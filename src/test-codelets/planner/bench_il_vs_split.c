/**
 * bench_il_vs_split.c -- Interleaved vs split-complex R=16 n1 benchmark
 *
 * Compares GENERATED IL codelet (4x4 CT, k_step=2) vs GENERATED split
 * codelet (4x4 CT, k_step=4). Same butterfly, different memory layout.
 *
 * Hypothesis: IL wins at high K (single memory stream, better spatial
 * locality), split wins at low K (k_step=4 vs 2, less loop overhead,
 * data fits in cache so bandwidth doesn't matter).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../bench_compat.h"

/* Split-complex R=16 n1 codelet (generated, k_step=4) */
#include "fft_radix16_avx2_ct_n1.h"

/* IL R=16 n1 codelet (generated, k_step=2) */
#include "fft_radix16_avx2_ct_n1_il.h"

#define R 16

/* ================================================================
 * Correctness: brute-force DFT-16 (split-complex reference)
 * ================================================================ */
static void bruteforce_dft16(const double *xr, const double *xi,
                             double *Xr, double *Xi, size_t K) {
    for (int m = 0; m < R; m++)
        for (size_t b = 0; b < K; b++) {
            double sr = 0, si = 0;
            for (int n = 0; n < R; n++) {
                double angle = -2.0 * M_PI * (double)n * (double)m / (double)R;
                sr += xr[n*K+b]*cos(angle) - xi[n*K+b]*sin(angle);
                si += xr[n*K+b]*sin(angle) + xi[n*K+b]*cos(angle);
            }
            Xr[m*K+b] = sr; Xi[m*K+b] = si;
        }
}

/* Convert split -> interleaved */
static void split_to_il(const double *re, const double *im, double *il,
                        int N, size_t K) {
    for (int n = 0; n < N; n++)
        for (size_t k = 0; k < K; k++) {
            il[(n*K+k)*2 + 0] = re[n*K+k];
            il[(n*K+k)*2 + 1] = im[n*K+k];
        }
}

/* Convert interleaved -> split */
static void il_to_split(const double *il, double *re, double *im,
                        int N, size_t K) {
    for (int n = 0; n < N; n++)
        for (size_t k = 0; k < K; k++) {
            re[n*K+k] = il[(n*K+k)*2 + 0];
            im[n*K+k] = il[(n*K+k)*2 + 1];
        }
}

/* Max absolute error */
static double max_err(const double *ar, const double *ai,
                      const double *br, const double *bi, size_t total) {
    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(ar[i] - br[i]);
        double ei = fabs(ai[i] - bi[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }
    return mx;
}

/* ================================================================
 * Correctness test
 * ================================================================ */
static int test_correctness(size_t K) {
    size_t total = (size_t)R * K;

    double *in_re  = (double*)aligned_alloc(64, total * sizeof(double));
    double *in_im  = (double*)aligned_alloc(64, total * sizeof(double));
    double *ref_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *ref_im = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_split_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_split_im = (double*)aligned_alloc(64, total * sizeof(double));
    double *in_il  = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    double *out_il = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    double *out_il_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_il_im = (double*)aligned_alloc(64, total * sizeof(double));

    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Reference */
    bruteforce_dft16(in_re, in_im, ref_re, ref_im, K);

    /* Split codelet: radix16_n1_fwd_avx2(in_re, in_im, out_re, out_im, is, os, vl) */
    radix16_n1_fwd_avx2(in_re, in_im, out_split_re, out_split_im, K, K, K);
    double err_split = max_err(ref_re, ref_im, out_split_re, out_split_im, total);

    /* IL codelet: radix16_n1_il_fwd_avx2(in, out, is, os, vl) */
    split_to_il(in_re, in_im, in_il, R, K);
    memset(out_il, 0, total * 2 * sizeof(double));
    radix16_n1_il_fwd_avx2(in_il, out_il, K, K, K);
    il_to_split(out_il, out_il_re, out_il_im, R, K);
    double err_il = max_err(ref_re, ref_im, out_il_re, out_il_im, total);

    printf("  K=%4zu: split err=%.2e, IL err=%.2e", K, err_split, err_il);
    int ok = (err_split < 1e-10 && err_il < 1e-10);
    printf("  %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        printf("    split[0] = (%f,%f) ref[0] = (%f,%f)\n",
               out_split_re[0], out_split_im[0], ref_re[0], ref_im[0]);
        printf("    il[0]    = (%f,%f)\n", out_il_re[0], out_il_im[0]);
    }

    aligned_free(in_re); aligned_free(in_im);
    aligned_free(ref_re); aligned_free(ref_im);
    aligned_free(out_split_re); aligned_free(out_split_im);
    aligned_free(in_il); aligned_free(out_il);
    aligned_free(out_il_re); aligned_free(out_il_im);
    return ok;
}

/* ================================================================
 * Benchmark helpers
 * ================================================================ */
static double bench_split_fwd(size_t K, int reps) {
    size_t total = (size_t)R * K;
    double *in_re  = (double*)aligned_alloc(64, total * sizeof(double));
    double *in_im  = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_im = (double*)aligned_alloc(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }
    for (int i = 0; i < 10; i++)
        radix16_n1_fwd_avx2(in_re, in_im, out_re, out_im, K, K, K);
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            radix16_n1_fwd_avx2(in_re, in_im, out_re, out_im, K, K, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    aligned_free(in_re); aligned_free(in_im);
    aligned_free(out_re); aligned_free(out_im);
    return best;
}

static double bench_split_bwd(size_t K, int reps) {
    size_t total = (size_t)R * K;
    double *in_re  = (double*)aligned_alloc(64, total * sizeof(double));
    double *in_im  = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_re = (double*)aligned_alloc(64, total * sizeof(double));
    double *out_im = (double*)aligned_alloc(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }
    for (int i = 0; i < 10; i++)
        radix16_n1_bwd_avx2(in_re, in_im, out_re, out_im, K, K, K);
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            radix16_n1_bwd_avx2(in_re, in_im, out_re, out_im, K, K, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    aligned_free(in_re); aligned_free(in_im);
    aligned_free(out_re); aligned_free(out_im);
    return best;
}

static double bench_il_fwd(size_t K, int reps) {
    size_t total = (size_t)R * K;
    double *in_il  = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    double *out_il = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    for (size_t i = 0; i < total * 2; i++)
        in_il[i] = (double)rand()/RAND_MAX - 0.5;
    for (int i = 0; i < 10; i++)
        radix16_n1_il_fwd_avx2(in_il, out_il, K, K, K);
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            radix16_n1_il_fwd_avx2(in_il, out_il, K, K, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    aligned_free(in_il); aligned_free(out_il);
    return best;
}

static double bench_il_bwd(size_t K, int reps) {
    size_t total = (size_t)R * K;
    double *in_il  = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    double *out_il = (double*)aligned_alloc(64, total * 2 * sizeof(double));
    for (size_t i = 0; i < total * 2; i++)
        in_il[i] = (double)rand()/RAND_MAX - 0.5;
    for (int i = 0; i < 10; i++)
        radix16_n1_il_bwd_avx2(in_il, out_il, K, K, K);
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            radix16_n1_il_bwd_avx2(in_il, out_il, K, K, K);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    aligned_free(in_il); aligned_free(out_il);
    return best;
}

int main(void) {
    srand(42);

    printf("VectorFFT: IL vs Split-Complex R=16 N1 Benchmark\n");
    printf("==================================================\n");
    printf("  Split: gen_radix16 ct_n1 (4x4 CT, k_step=4, 2 buffers)\n");
    printf("  IL:    gen_radix16 ct_n1_il (4x4 CT, k_step=2, 1 buffer)\n");
    printf("  Same butterfly, different memory layout.\n");
    printf("\n");

    /* Correctness */
    printf("Correctness checks:\n");
    int ok = 1;
    size_t test_Ks[] = {4, 8, 16, 64, 256};
    for (int i = 0; i < 5; i++)
        ok &= test_correctness(test_Ks[i]);
    if (!ok) {
        printf("\n*** CORRECTNESS FAILURE — aborting bench ***\n");
        return 1;
    }
    printf("\n");

    /* Benchmark */
    size_t Ks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    printf("%-6s | %10s %10s %7s | %10s %10s %7s | %8s\n",
           "K", "split_fwd", "il_fwd", "ratio", "split_bwd", "il_bwd", "ratio", "winner");
    printf("%-6s-+-%10s-%10s-%7s-+-%10s-%10s-%7s-+-%8s\n",
           "------", "----------", "----------", "-------",
           "----------", "----------", "-------", "--------");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)R * K;

        int reps = (int)(2e6 / (total + 1));
        if (reps < 50) reps = 50;
        if (reps > 200000) reps = 200000;

        double sf = bench_split_fwd(K, reps);
        double ilf = bench_il_fwd(K, reps);
        double sb = bench_split_bwd(K, reps);
        double ilb = bench_il_bwd(K, reps);

        double ratio_f = ilf / sf;  /* >1 = split wins, <1 = IL wins */
        double ratio_b = ilb / sb;

        const char *winner;
        double avg_ratio = (ratio_f + ratio_b) / 2.0;
        if (avg_ratio < 0.95) winner = "IL";
        else if (avg_ratio > 1.05) winner = "SPLIT";
        else winner = "~TIE";

        printf("%-6zu | %8.1f ns %8.1f ns %6.2fx | %8.1f ns %8.1f ns %6.2fx | %8s\n",
               K, sf, ilf, ratio_f, sb, ilb, ratio_b, winner);
    }

    printf("\n");
    printf("ratio = IL_time / split_time  (>1 = split faster, <1 = IL faster)\n");
    printf("Split: k_step=4, 2 memory streams (re[] + im[])\n");
    printf("IL:    k_step=2, 1 memory stream ([re,im,re,im,...])\n");

    return 0;
}
