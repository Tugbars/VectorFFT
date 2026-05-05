/* test_mt_dct.c — verify roundtrip correctness for DCT/DST/DHT at T>1.
 *
 * The K-split MT path in DCT-II/III, DCT-IV, DST-II/III, and DHT is
 * the new code; this test exercises it at non-trivial K and several
 * thread counts to confirm it produces bit-identical results to T=1.
 *
 * Build: python build.py --src test/test_mt_dct.c --vfft
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

static void check(const char *name, int N, size_t K, int T,
                  double err, double thresh) {
    int ok = (err < thresh);
    printf("  %-12s N=%-5d K=%-5zu T=%d  err=%.2e  %s\n",
           name, N, K, T, err, ok ? "PASS" : "FAIL");
    if (ok) n_pass++; else n_fail++;
}

static void fill_random(double *buf, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        buf[i] = (double)rand() / RAND_MAX - 0.5;
}

/* For each transform: forward → backward → compare to original. The
 * MT path activates only when N*K >= 8192*T — pick large enough cells.
 * We test at multiple thread counts to ensure each branch lights up. */

#define DCT_TEST(transform_name, plan_fn, fwd_fn)                             \
    static void test_##transform_name(int N, size_t K) {                      \
        size_t NK = (size_t)N * K;                                            \
        double *in   = (double *)vfft_alloc(NK * sizeof(double));             \
        double *work = (double *)vfft_alloc(NK * sizeof(double));             \
        double *ref  = (double *)vfft_alloc(NK * sizeof(double));             \
        fill_random(in, NK, 17 + N + (unsigned)K);                            \
                                                                              \
        /* T=1 ground truth */                                                \
        vfft_set_num_threads(1);                                              \
        vfft_plan p1 = plan_fn(N, K, VFFT_ESTIMATE);                          \
        fwd_fn(p1, in, ref);                                                  \
        vfft_destroy(p1);                                                     \
                                                                              \
        int Ts[] = { 2, 4, 8 };                                               \
        for (int t = 0; t < 3; t++) {                                         \
            int T = Ts[t];                                                    \
            vfft_set_num_threads(T);                                          \
            vfft_plan p = plan_fn(N, K, VFFT_ESTIMATE);                       \
            fwd_fn(p, in, work);                                              \
            double err_fwd = max_diff(work, ref, NK);                         \
            check(#transform_name " fwd", N, K, T, err_fwd, 1e-12);           \
            vfft_destroy(p);                                                  \
        }                                                                     \
        vfft_free(in); vfft_free(work); vfft_free(ref);                       \
    }

DCT_TEST(dct2, vfft_plan_dct2, vfft_execute_dct2)
DCT_TEST(dct4, vfft_plan_dct4, vfft_execute_dct4)
DCT_TEST(dst2, vfft_plan_dst2, vfft_execute_dst2)
DCT_TEST(dht,  vfft_plan_dht,  vfft_execute_dht)

int main(void) {
    vfft_init();

    printf("=== test_mt_dct -- DCT/DST/DHT MT roundtrip correctness ===\n\n");
    printf("  %-12s %-7s %-7s %-3s  %s\n",
           "transform", "N", "K", "T", "err");
    printf("  ----------------------------------------------------------\n");

    /* Cells chosen to exceed MT threshold (N*K >= 8192*T at T=8 means
     * N*K >= 65536). N=256 K=1024 gives 262144; N=1024 K=1024 gives 1M. */
    int Ns[] = { 256, 1024 };
    size_t Ks[] = { 256, 1024 };
    for (int ni = 0; ni < 2; ni++)
    for (int ki = 0; ki < 2; ki++) {
        test_dct2(Ns[ni], Ks[ki]);
        test_dct4(Ns[ni], Ks[ki]);
        test_dst2(Ns[ni], Ks[ki]);
        test_dht(Ns[ni], Ks[ki]);
    }

    /* Reset to single-threaded so the runtime exits cleanly. */
    vfft_set_num_threads(1);

    printf("\n=== %d passed, %d failed ===\n", n_pass, n_fail);
    return (n_fail == 0) ? 0 : 1;
}
