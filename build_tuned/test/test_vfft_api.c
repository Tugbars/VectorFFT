/* test_vfft_api.c -- smoke test for the public vfft.h API.
 *
 * Exercises every plan type via vfft.h alone (no core/* internals),
 * roundtrips each, prints PASS/FAIL.
 *
 * Build: python build.py --src test/test_vfft_api.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vfft.h"

static int n_pass = 0, n_fail = 0;

static double max_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void check(const char *name, double err, double thresh) {
    int ok = (err < thresh);
    printf("  %-30s err=%.2e  %s\n", name, err, ok ? "PASS" : "FAIL");
    if (ok) n_pass++; else n_fail++;
}

/* ─── 1D C2C ────────────────────────────────────────────────────── */
static void test_c2c(void) {
    int N = 256; size_t K = 32; size_t NK = (size_t)N * K;
    double *re = (double *)vfft_alloc(NK * sizeof(double));
    double *im = (double *)vfft_alloc(NK * sizeof(double));
    double *re0 = (double *)vfft_alloc(NK * sizeof(double));
    double *im0 = (double *)vfft_alloc(NK * sizeof(double));
    srand(1);
    for (size_t i = 0; i < NK; i++) { re[i] = re0[i] = (double)rand()/RAND_MAX - 0.5;
                                       im[i] = im0[i] = (double)rand()/RAND_MAX - 0.5; }

    vfft_plan p = vfft_plan_c2c(N, K, VFFT_ESTIMATE);
    vfft_execute_fwd(p, re, im);
    vfft_execute_bwd_normalized(p, re, im);
    double err = max_diff(re, re0, NK) + max_diff(im, im0, NK);
    check("vfft_plan_c2c (ESTIMATE)", err, 1e-12);

    vfft_destroy(p);
    vfft_free(re); vfft_free(im); vfft_free(re0); vfft_free(im0);
}

/* ─── 1D R2C / C2R ──────────────────────────────────────────────── */
static void test_r2c(void) {
    int N = 256; size_t K = 32;
    size_t real_sz = (size_t)N * K;
    size_t cplx_sz = (size_t)(N/2+1) * K;
    double *real_in = (double *)vfft_alloc(real_sz * sizeof(double));
    /* out_re must be N*K (used as workspace); only the lower (N/2+1)*K is valid output.
     * See vfft.h note on vfft_execute_r2c. */
    double *out_re  = (double *)vfft_alloc(real_sz * sizeof(double));
    double *out_im  = (double *)vfft_alloc(cplx_sz * sizeof(double));
    double *back    = (double *)vfft_alloc(real_sz * sizeof(double));
    double *orig    = (double *)vfft_alloc(real_sz * sizeof(double));
    srand(2);
    for (size_t i = 0; i < real_sz; i++) real_in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_r2c(N, K, VFFT_ESTIMATE);
    vfft_execute_r2c(p, real_in, out_re, out_im);
    vfft_execute_c2r(p, out_re, out_im, back);
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < real_sz; i++) back[i] *= inv_N;
    check("vfft_plan_r2c (ESTIMATE)", max_diff(orig, back, real_sz), 1e-12);

    vfft_destroy(p);
    vfft_free(real_in); vfft_free(out_re); vfft_free(out_im);
    vfft_free(back); vfft_free(orig);
}

/* ─── 2D C2C ────────────────────────────────────────────────────── */
static void test_2d(void) {
    int N1 = 64, N2 = 64; size_t NK = (size_t)N1 * N2;
    double *re = (double *)vfft_alloc(NK * sizeof(double));
    double *im = (double *)vfft_alloc(NK * sizeof(double));
    double *re0 = (double *)vfft_alloc(NK * sizeof(double));
    double *im0 = (double *)vfft_alloc(NK * sizeof(double));
    srand(3);
    for (size_t i = 0; i < NK; i++) { re[i] = re0[i] = (double)rand()/RAND_MAX - 0.5;
                                       im[i] = im0[i] = (double)rand()/RAND_MAX - 0.5; }

    vfft_plan p = vfft_plan_2d(N1, N2, VFFT_ESTIMATE);
    vfft_execute_fwd(p, re, im);
    vfft_execute_bwd(p, re, im);
    double inv = 1.0 / (double)NK;
    for (size_t i = 0; i < NK; i++) { re[i] *= inv; im[i] *= inv; }
    double err = max_diff(re, re0, NK) + max_diff(im, im0, NK);
    check("vfft_plan_2d (ESTIMATE)", err, 1e-12);

    vfft_destroy(p);
    vfft_free(re); vfft_free(im); vfft_free(re0); vfft_free(im0);
}

/* ─── 2D R2C / C2R ──────────────────────────────────────────────── */
static void test_2d_r2c(void) {
    int N1 = 64, N2 = 64;
    size_t real_sz = (size_t)N1 * N2;
    size_t cplx_sz = (size_t)N1 * (N2/2+1);
    double *real_in = (double *)vfft_alloc(real_sz * sizeof(double));
    double *out_re  = (double *)vfft_alloc(cplx_sz * sizeof(double));
    double *out_im  = (double *)vfft_alloc(cplx_sz * sizeof(double));
    double *back    = (double *)vfft_alloc(real_sz * sizeof(double));
    double *orig    = (double *)vfft_alloc(real_sz * sizeof(double));
    srand(4);
    for (size_t i = 0; i < real_sz; i++) real_in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_2d_r2c(N1, N2, VFFT_ESTIMATE);
    vfft_execute_2d_r2c(p, real_in, out_re, out_im);
    vfft_execute_2d_c2r(p, out_re, out_im, back);
    double inv = 1.0 / (double)real_sz;
    for (size_t i = 0; i < real_sz; i++) back[i] *= inv;
    check("vfft_plan_2d_r2c (ESTIMATE)", max_diff(orig, back, real_sz), 1e-12);

    vfft_destroy(p);
    vfft_free(real_in); vfft_free(out_re); vfft_free(out_im);
    vfft_free(back); vfft_free(orig);
}

/* ─── DCT-II / DCT-III roundtrip ────────────────────────────────── */
static void test_dct23(void) {
    int N = 64; size_t K = 32; size_t NK = (size_t)N * K;
    double *in   = (double *)vfft_alloc(NK * sizeof(double));
    double *coef = (double *)vfft_alloc(NK * sizeof(double));
    double *back = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    srand(5);
    for (size_t i = 0; i < NK; i++) in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_dct2(N, K, VFFT_ESTIMATE);
    vfft_execute_dct2(p, in, coef);
    vfft_execute_dct3(p, coef, back);
    double inv = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < NK; i++) back[i] *= inv;
    check("vfft_plan_dct2 (ESTIMATE)", max_diff(orig, back, NK), 1e-12);

    vfft_destroy(p);
    vfft_free(in); vfft_free(coef); vfft_free(back); vfft_free(orig);
}

/* ─── DCT-IV roundtrip (involutory) ─────────────────────────────── */
static void test_dct4(void) {
    int N = 64; size_t K = 32; size_t NK = (size_t)N * K;
    double *in   = (double *)vfft_alloc(NK * sizeof(double));
    double *out  = (double *)vfft_alloc(NK * sizeof(double));
    double *back = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    srand(6);
    for (size_t i = 0; i < NK; i++) in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_dct4(N, K, VFFT_ESTIMATE);
    vfft_execute_dct4(p, in, out);
    vfft_execute_dct4(p, out, back);
    double inv = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < NK; i++) back[i] *= inv;
    check("vfft_plan_dct4 (ESTIMATE)", max_diff(orig, back, NK), 1e-12);

    vfft_destroy(p);
    vfft_free(in); vfft_free(out); vfft_free(back); vfft_free(orig);
}

/* ─── DST-II / DST-III roundtrip ────────────────────────────────── */
static void test_dst23(void) {
    int N = 64; size_t K = 32; size_t NK = (size_t)N * K;
    double *in   = (double *)vfft_alloc(NK * sizeof(double));
    double *coef = (double *)vfft_alloc(NK * sizeof(double));
    double *back = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    srand(7);
    for (size_t i = 0; i < NK; i++) in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_dst2(N, K, VFFT_ESTIMATE);
    vfft_execute_dst2(p, in, coef);
    vfft_execute_dst3(p, coef, back);
    double inv = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < NK; i++) back[i] *= inv;
    check("vfft_plan_dst2 (ESTIMATE)", max_diff(orig, back, NK), 1e-12);

    vfft_destroy(p);
    vfft_free(in); vfft_free(coef); vfft_free(back); vfft_free(orig);
}

/* ─── DHT (self-inverse) ────────────────────────────────────────── */
static void test_dht(void) {
    int N = 64; size_t K = 32; size_t NK = (size_t)N * K;
    double *in   = (double *)vfft_alloc(NK * sizeof(double));
    double *out  = (double *)vfft_alloc(NK * sizeof(double));
    double *back = (double *)vfft_alloc(NK * sizeof(double));
    double *orig = (double *)vfft_alloc(NK * sizeof(double));
    srand(8);
    for (size_t i = 0; i < NK; i++) in[i] = orig[i] = (double)rand()/RAND_MAX - 0.5;

    vfft_plan p = vfft_plan_dht(N, K, VFFT_ESTIMATE);
    vfft_execute_dht(p, in, out);
    vfft_execute_dht(p, out, back);
    double inv = 1.0 / (double)N;
    for (size_t i = 0; i < NK; i++) back[i] *= inv;
    check("vfft_plan_dht (ESTIMATE)", max_diff(orig, back, NK), 1e-12);

    vfft_destroy(p);
    vfft_free(in); vfft_free(out); vfft_free(back); vfft_free(orig);
}

/* ─── Plan-creation flag exercise ───────────────────────────────
 * Verifies WISDOM_ONLY misses cleanly, load/MEASURE hit, save round-trips. */
static void test_flags(void) {
    int N = 64; size_t K = 32;

    /* Empty wisdom: WISDOM_ONLY should return NULL. */
    vfft_forget_wisdom();
    vfft_plan p = vfft_plan_c2c(N, K, VFFT_WISDOM_ONLY);
    if (p == NULL) {
        printf("  %-30s WISDOM_ONLY+empty -> NULL  PASS\n", "vfft_plan_c2c flags");
        n_pass++;
    } else {
        printf("  %-30s WISDOM_ONLY+empty -> non-NULL  FAIL\n", "vfft_plan_c2c flags");
        n_fail++;
        vfft_destroy(p);
    }

    /* Load the dev wisdom (build_tuned/vfft_wisdom_tuned.txt — CWD-relative). */
    int rc = vfft_load_wisdom("vfft_wisdom_tuned.txt");
    printf("  %-30s vfft_load_wisdom rc=%d %s\n",
           "vfft_load_wisdom",
           rc, rc == 0 ? "PASS" : "(no wisdom file — ok)");
    if (rc == 0) n_pass++;  /* file exists, load succeeded */

    /* MEASURE with loaded wisdom should hit (or calibrate quickly) and produce a plan. */
    p = vfft_plan_c2c(N, K, VFFT_MEASURE);
    if (p) {
        printf("  %-30s MEASURE -> plan  PASS\n", "vfft_plan_c2c flags");
        n_pass++;
        vfft_destroy(p);
    } else {
        printf("  %-30s MEASURE -> NULL  FAIL\n", "vfft_plan_c2c flags");
        n_fail++;
    }

    /* Save round-trip — write current in-memory wisdom to a temp file. */
    rc = vfft_save_wisdom("vfft_wisdom_tmp.txt");
    printf("  %-30s vfft_save_wisdom rc=%d %s\n",
           "vfft_save_wisdom",
           rc, rc == 0 ? "PASS" : "FAIL");
    if (rc == 0) n_pass++; else n_fail++;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_init();
    vfft_set_num_threads(1);

    fprintf(stderr, "=== test_vfft_api -- public API smoke test ===\n");
    fprintf(stderr, "  version: %s   ISA: %s\n\n", vfft_version(), vfft_isa());
    fflush(stderr);

    fprintf(stderr, "[c2c]\n"); fflush(stderr);     test_c2c();
    fprintf(stderr, "[r2c]\n"); fflush(stderr);     test_r2c();
    fprintf(stderr, "[2d]\n"); fflush(stderr);      test_2d();
    fprintf(stderr, "[2d_r2c]\n"); fflush(stderr);  test_2d_r2c();
    fprintf(stderr, "[dct23]\n"); fflush(stderr);   test_dct23();
    fprintf(stderr, "[dct4]\n"); fflush(stderr);    test_dct4();
    fprintf(stderr, "[dst23]\n"); fflush(stderr);   test_dst23();
    fprintf(stderr, "[dht]\n"); fflush(stderr);     test_dht();
    fprintf(stderr, "[flags]\n"); fflush(stderr);   test_flags();

    fprintf(stderr, "\n=== %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail;
}
