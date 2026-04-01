/**
 * bench_odd_radix.c -- Odd-radix AVX2 codelet benchmark vs FFTW_MEASURE
 *
 * Currently: R=5 (notw, dit_flat, dit_log3)
 * Add R=3, R=7 by including their headers and adding bench_radix() calls.
 *
 * Verifies correctness against FFTW before recording timing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

/* ── R=5 codelets: standard ── */
#include "fft_radix5_avx2_notw.h"
#include "fft_radix5_avx2_dit_tw.h"
#include "fft_radix5_avx2_dit_tw_log3.h"

/* CT codelets (n1_ovs + t1_dit) deferred to production executor integration.
 * Odd-R CT requires mixed radixes: e.g. pow2_n1_ovs + radix5_t1_dit.
 * The standalone codelet bench above validates the butterfly math. */

/* ── R=3 codelets ── */
#include "fft_radix3_avx2_notw.h"
#include "fft_radix3_avx2_dit_tw.h"
#include "fft_radix3_avx2_dit_tw_log3.h"

/* ── R=7 codelets: Sethi-Ullman scheduled, gen_radix7.py ── */
#include "fft_radix7_avx2_notw.h"
#include "fft_radix7_avx2_dit_tw.h"
#include "fft_radix7_avx2_dit_tw_log3.h"

/* ── R=25 codelets: 5x5 CT, gen_radix25.py ── */
#include "fft_radix25_avx2_notw.h"
#include "fft_radix25_avx2_dit_tw.h"
#include "fft_radix25_avx2_dit_tw_log3.h"

/* ── R=11 codelets: genfft DAG, gen_radix11.py ── */
#include "fft_radix11_avx2_notw.h"
#include "fft_radix11_avx2_dit_tw.h"
#include "fft_radix11_avx2_dit_tw_log3.h"

/* ── Function types ── */
typedef void (*notw_fn)(const double*, const double*, double*, double*, size_t);
typedef void (*tw_fn)(const double*, const double*, double*, double*,
                      const double*, const double*, size_t);
/* ── Twiddle init for standard codelets: W_N^{n*k}, layout tw[(n-1)*K+k] ── */
static void init_tw(double *twr, double *twi, int R, size_t K) {
    size_t N = (size_t)R * K;
    for (int n = 1; n < R; n++) {
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * n * k / (double)N;
            twr[(n - 1) * K + k] = cos(a);
            twi[(n - 1) * K + k] = sin(a);
        }
    }
}

/* ── Benchmark helpers ── */
static double bench_nf(notw_fn fn, const double *ir, const double *ii,
                       double *or_, double *oi, size_t K, int reps) {
    for (int i = 0; i < 20; i++) fn(ir, ii, or_, oi, K);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(ir, ii, or_, oi, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static double bench_tf(tw_fn fn, const double *ir, const double *ii,
                       double *or_, double *oi,
                       const double *twr, const double *twi,
                       size_t K, int reps) {
    for (int i = 0; i < 20; i++) fn(ir, ii, or_, oi, twr, twi, K);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(ir, ii, or_, oi, twr, twi, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static double bench_fftw(int R, size_t K, int reps) {
    size_t N = (size_t)R * K;
    double *ri = fftw_malloc(N * 8), *ii = fftw_malloc(N * 8);
    double *ro = fftw_malloc(N * 8), *io = fftw_malloc(N * 8);
    for (size_t i = 0; i < N; i++) {
        ri[i] = (double)rand() / RAND_MAX - 0.5;
        ii[i] = (double)rand() / RAND_MAX - 0.5;
    }
    fftw_iodim dim  = { .n = R, .is = (int)K, .os = (int)K };
    fftw_iodim howm = { .n = (int)K, .is = 1, .os = 1 };
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm,
                                           ri, ii, ro, io, FFTW_MEASURE);
    if (!p) {
        fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io);
        return -1;
    }
    for (size_t i = 0; i < N; i++) {
        ri[i] = (double)rand() / RAND_MAX - 0.5;
        ii[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int i = 0; i < 20; i++) fftw_execute(p);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            fftw_execute_split_dft(p, ri, ii, ro, io);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    fftw_destroy_plan(p);
    fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io);
    return best;
}

/* ── Correctness: compare codelet output to FFTW_ESTIMATE ── */
static int verify_notw(int R, notw_fn fn, size_t K) {
    size_t N = (size_t)R * K;
    double *ir  = aligned_alloc(32, N * 8), *ii_ = aligned_alloc(32, N * 8);
    double *or_v = aligned_alloc(32, N * 8), *oi_v = aligned_alloc(32, N * 8);
    double *or_f = fftw_malloc(N * 8), *oi_f = fftw_malloc(N * 8);

    for (size_t i = 0; i < N; i++) {
        ir[i]  = (double)rand() / RAND_MAX - 0.5;
        ii_[i] = (double)rand() / RAND_MAX - 0.5;
    }

    fftw_iodim dim  = { .n = R, .is = (int)K, .os = (int)K };
    fftw_iodim howm = { .n = (int)K, .is = 1, .os = 1 };
    double *ir2 = fftw_malloc(N * 8), *ii2 = fftw_malloc(N * 8);
    memcpy(ir2, ir, N * 8); memcpy(ii2, ii_, N * 8);
    fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &howm,
                                           ir2, ii2, or_f, oi_f, FFTW_ESTIMATE);
    fftw_execute_split_dft(p, ir, ii_, or_f, oi_f);
    fftw_destroy_plan(p);
    fftw_free(ir2); fftw_free(ii2);

    fn(ir, ii_, or_v, oi_v, K);

    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double er = fabs(or_v[i] - or_f[i]);
        double ei = fabs(oi_v[i] - oi_f[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    aligned_free(ir); aligned_free(ii_);
    aligned_free(or_v); aligned_free(oi_v);
    fftw_free(or_f); fftw_free(oi_f);
    return max_err < 1e-10;
}

static int verify_tw(int R, tw_fn fn, notw_fn fn_notw, size_t K) {
    /*
     * The twiddled codelet computes: twiddle inputs, then butterfly.
     * FFTW computes a plain DFT-R (no twiddles), so we can't compare directly.
     * Instead: manually twiddle inputs, run notw, and compare against
     * the twiddled codelet's output.
     */
    size_t N = (size_t)R * K;
    double *ir  = aligned_alloc(32, N * 8), *ii_ = aligned_alloc(32, N * 8);
    double *or_tw = aligned_alloc(32, N * 8), *oi_tw = aligned_alloc(32, N * 8);
    double *ir_m  = aligned_alloc(32, N * 8), *ii_m = aligned_alloc(32, N * 8);
    double *or_ref = aligned_alloc(32, N * 8), *oi_ref = aligned_alloc(32, N * 8);
    double *twr = aligned_alloc(32, (R - 1) * K * 8);
    double *twi = aligned_alloc(32, (R - 1) * K * 8);

    for (size_t i = 0; i < N; i++) {
        ir[i]  = (double)rand() / RAND_MAX - 0.5;
        ii_[i] = (double)rand() / RAND_MAX - 0.5;
    }
    init_tw(twr, twi, R, K);

    /* Reference: manually twiddle inputs, then run notw */
    memcpy(ir_m, ir, N * 8);
    memcpy(ii_m, ii_, N * 8);
    for (int n = 1; n < R; n++) {
        for (size_t k = 0; k < K; k++) {
            double wr = twr[(n - 1) * K + k];
            double wi = twi[(n - 1) * K + k];
            double xr = ir_m[n * K + k], xi = ii_m[n * K + k];
            ir_m[n * K + k] = xr * wr - xi * wi;
            ii_m[n * K + k] = xr * wi + xi * wr;
        }
    }
    fn_notw(ir_m, ii_m, or_ref, oi_ref, K);

    /* Our twiddled codelet */
    fn(ir, ii_, or_tw, oi_tw, twr, twi, K);

    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double er = fabs(or_tw[i] - or_ref[i]);
        double ei = fabs(oi_tw[i] - oi_ref[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    aligned_free(ir); aligned_free(ii_);
    aligned_free(or_tw); aligned_free(oi_tw);
    aligned_free(ir_m); aligned_free(ii_m);
    aligned_free(or_ref); aligned_free(oi_ref);
    aligned_free(twr); aligned_free(twi);
    return max_err < 1e-10;
}

/* ── Generic radix benchmark: correctness + perf table ── */
static int bench_radix(int R, const char *label,
                       notw_fn fn_notw, tw_fn fn_flat, tw_fn fn_log3) {
    printf("\n== R=%d %s ==\n\n", R, label);

    /* Correctness */
    printf("Correctness:\n");
    int ok = 1;
    size_t test_Ks[] = { 4, 8, 16, 32, 64, 128 };
    int n_test = (int)(sizeof(test_Ks) / sizeof(test_Ks[0]));
    for (int i = 0; i < n_test; i++) {
        size_t K = test_Ks[i];
        int v_notw = verify_notw(R, fn_notw, K);
        int v_flat = verify_tw(R, fn_flat, fn_notw, K);
        int v_log3 = fn_log3 ? verify_tw(R, fn_log3, fn_notw, K) : 1;
        printf("  K=%-4zu  notw=%s  flat=%s  log3=%s\n", K,
               v_notw ? "OK" : "FAIL", v_flat ? "OK" : "FAIL",
               fn_log3 ? (v_log3 ? "OK" : "FAIL") : "n/a");
        if (!v_notw || !v_flat || !v_log3) ok = 0;
    }
    if (!ok) {
        printf("  *** CORRECTNESS FAILURE ***\n");
        return 1;
    }
    printf("  All correct.\n\n");

    /* Performance */
    size_t Ks[] = { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));

    if (fn_log3) {
        printf("%-5s %-7s %8s  %14s %14s %14s\n",
               "K", "N", "FFTW_M", "notw", "dit_flat", "dit_log3");
        printf("%-5s %-7s %8s  %14s %14s %14s\n",
               "-----", "-------", "--------",
               "--------------", "--------------", "--------------");
    } else {
        printf("%-5s %-7s %8s  %14s %14s\n",
               "K", "N", "FFTW_M", "notw", "dit_flat");
        printf("%-5s %-7s %8s  %14s %14s\n",
               "-----", "-------", "--------",
               "--------------", "--------------");
    }

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki], N = (size_t)R * K;

        double *ir  = aligned_alloc(32, N * 8);
        double *ii_ = aligned_alloc(32, N * 8);
        double *or_ = aligned_alloc(32, N * 8);
        double *oi  = aligned_alloc(32, N * 8);
        double *twr = aligned_alloc(32, (R - 1) * K * 8);
        double *twi = aligned_alloc(32, (R - 1) * K * 8);

        for (size_t i = 0; i < N; i++) {
            ir[i]  = (double)rand() / RAND_MAX - 0.5;
            ii_[i] = (double)rand() / RAND_MAX - 0.5;
        }
        init_tw(twr, twi, R, K);

        int reps = (int)(2e6 / (N + 1));
        if (reps < 200) reps = 200;
        if (reps > 2000000) reps = 2000000;

        double fftw_ns = bench_fftw(R, K, reps);

        double ns_notw = bench_nf(fn_notw, ir, ii_, or_, oi, K, reps);
        double ns_flat = bench_tf(fn_flat, ir, ii_, or_, oi, twr, twi, K, reps);

        printf("%-5zu %-7zu %8.1f", K, N, fftw_ns);
        printf("  %5.0f(%5.2fx)", ns_notw,
               fftw_ns > 0 ? fftw_ns / ns_notw : 0);
        printf("  %5.0f(%5.2fx)", ns_flat,
               fftw_ns > 0 ? fftw_ns / ns_flat : 0);

        if (fn_log3) {
            double ns_log3 = bench_tf(fn_log3, ir, ii_, or_, oi,
                                      twr, twi, K, reps);
            printf("  %5.0f(%5.2fx)", ns_log3,
                   fftw_ns > 0 ? fftw_ns / ns_log3 : 0);
        }

        printf("\n");

        aligned_free(ir); aligned_free(ii_);
        aligned_free(or_); aligned_free(oi);
        aligned_free(twr); aligned_free(twi);
    }

    return 0;
}

/* ── Main ── */
int main(void) {
    srand(42);
    printf("VectorFFT Odd-Radix AVX2 Benchmark vs FFTW_MEASURE\n");

    int fail = 0;

    /* R=5 */
    fail |= bench_radix(5, "AVX2",
        (notw_fn)radix5_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix5_tw_flat_dit_kernel_fwd_avx2,
        (tw_fn)radix5_tw_log3_dit_kernel_fwd_avx2);

    /* R=3 */
    fail |= bench_radix(3, "AVX2",
        (notw_fn)radix3_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix3_tw_flat_dit_kernel_fwd_avx2,
        (tw_fn)radix3_tw_log3_dit_kernel_fwd_avx2);

    /* R=7: Sethi-Ullman scheduled, flat + log3 */
    fail |= bench_radix(7, "AVX2",
        (notw_fn)radix7_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix7_tw_flat_dit_kernel_fwd_avx2,
        (tw_fn)radix7_tw_log3_dit_kernel_fwd_avx2);

    /* R=25: 5x5 CT codelet */
    fail |= bench_radix(25, "AVX2 (5x5 CT)",
        (notw_fn)radix25_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix25_tw_flat_dit_kernel_fwd_avx2,
        (tw_fn)radix25_tw_log3_dit_kernel_fwd_avx2);

    /* R=11: genfft DAG codelet */
    fail |= bench_radix(11, "AVX2 (genfft DAG)",
        (notw_fn)radix11_n1_dit_kernel_fwd_avx2,
        (tw_fn)radix11_tw_flat_dit_kernel_fwd_avx2,
        (tw_fn)radix11_tw_log3_dit_kernel_fwd_avx2);

    /* CT end-to-end (n1_ovs + t1_dit) deferred to production executor.
     * Odd-R CT requires mixed radixes: e.g. pow2_n1_ovs + radix5_t1_dit.
     * Same-radix CT only works for N=R² which isn't SIMD-aligned for R=5. */

    if (fail) {
        printf("\n*** SOME TESTS FAILED ***\n");
        return 1;
    }

    printf("\nDone.\n");
    return 0;
}
