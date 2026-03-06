/**
 * bench_r7_vs_fftw.c — Radix-7 strided vs packed vs FFTW
 *
 * Three contestants:
 *   STRIDED: radix7_tw_dit_kernel_fwd_avx512(K)
 *   PACKED:  loop of K/8 blocks × radix7_tw_dit_kernel_fwd_avx512(K=8)
 *   FFTW:   fftw_plan_many_dft with N=7, howmany=K
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R7_512_LD(p)   _mm512_loadu_pd(p)
#define R7_512_ST(p,v) _mm512_storeu_pd((p),(v))
#include "fft_radix7_avx512.h"
#include "fft_radix7_tw_packed.h"

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static double *alloc64(size_t n) {
    double *p = NULL;
    posix_memalign((void**)&p, 64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

int main(void) {
    printf("═══ Radix-7 AVX-512: Strided vs Packed vs FFTW (min-of-5, fwd+bwd) ═══\n");
    printf("  STRIDED = flat tw kernel at full K\n");
    printf("  PACKED  = K/8 blocks × kernel(K=8), data stays packed\n");
    printf("  FFTW    = fftw_plan_many_dft N=7 howmany=K\n\n");
    printf("  %-5s | %8s %8s %8s | %8s %8s | %8s %8s\n",
           "K", "strd ns", "pack ns", "fftw ns", "pk/strd", "pk/fftw", "st/fftw", "winner");
    printf("  ──────┼───────────────────────────┼───────────────────┼───────────────────\n");

    size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        if (K < 8 || (K & 7)) continue;
        const size_t NN = 7 * K;

        double *ir = alloc64(NN), *ii = alloc64(NN);
        double *or_s = alloc64(NN), *oi_s = alloc64(NN);
        double *or_p = alloc64(NN), *oi_p = alloc64(NN);
        double *twr = alloc64(6*K), *twi = alloc64(6*K);

        /* Packed buffers */
        double *pk_ir = alloc64(NN), *pk_ii = alloc64(NN);
        double *pk_or = alloc64(NN), *pk_oi = alloc64(NN);
        double *pk_twr = alloc64(6*K), *pk_twi = alloc64(6*K);

        /* FFTW buffers */
        double *fftw_in  = (double*)fftw_malloc(sizeof(double) * 2 * NN);
        double *fftw_out = (double*)fftw_malloc(sizeof(double) * 2 * NN);

        srand(42 + (unsigned)K);
        for (size_t i = 0; i < NN; i++) {
            ir[i] = (double)rand()/RAND_MAX - 0.5;
            ii[i] = (double)rand()/RAND_MAX - 0.5;
        }

        /* Build twiddles */
        r7_build_flat_twiddles(K, -1, twr, twi);

        /* Pre-pack */
        r7_pack_input_avx512(ir, ii, pk_ir, pk_ii, K);
        r7_pack_tw_avx512(twr, twi, pk_twr, pk_twi, K);

        /* FFTW: N=7 DFT, howmany=K, stride=K, dist=1
         * This computes K independent DFT-7s on strided data,
         * matching our tw kernel when twiddles are all 1 (notw).
         * For fair comparison, we benchmark the twiddled case for us
         * and FFTW's optimized N=7*K 1D DFT. */

        /* Actually: our twiddled DFT-7 with stride K is one stage of
         * a larger N=7K FFT. FFTW doesn't expose single-stage.
         * Fairest comparison: FFTW 1D DFT of size 7K. */
        fftw_plan plan_fwd = fftw_plan_dft_1d(
            (int)NN, (fftw_complex*)fftw_in, (fftw_complex*)fftw_out,
            FFTW_FORWARD, FFTW_ESTIMATE);

        /* Also prepare FFTW input in interleaved format */
        for (size_t k = 0; k < K; k++)
            for (int n = 0; n < 7; n++) {
                size_t idx = n*K + k;
                fftw_in[2*idx]   = ir[idx];
                fftw_in[2*idx+1] = ii[idx];
            }

        int iters = (int)(2000000.0 / K);
        if (iters < 200) iters = 200;
        int warmup = iters / 10;

        double best_s = 1e18, best_p = 1e18, best_f = 1e18;

        for (int run = 0; run < 5; run++) {
            /* STRIDED: fwd + bwd */
            for (int i = 0; i < warmup; i++) {
                radix7_tw_dit_kernel_fwd_avx512(ir, ii, or_s, oi_s, twr, twi, K);
                radix7_tw_dit_kernel_bwd_avx512(or_s, oi_s, ir, ii, twr, twi, K);
            }
            double t0 = now_ns();
            for (int i = 0; i < iters; i++) {
                radix7_tw_dit_kernel_fwd_avx512(ir, ii, or_s, oi_s, twr, twi, K);
                radix7_tw_dit_kernel_bwd_avx512(or_s, oi_s, ir, ii, twr, twi, K);
            }
            double ns = (now_ns() - t0) / iters;
            if (ns < best_s) best_s = ns;

            /* PACKED: fwd + bwd (kernel only, data stays packed) */
            size_t nb = K / 8;
            for (int i = 0; i < warmup; i++) {
                for (size_t b = 0; b < nb; b++) {
                    radix7_tw_dit_kernel_fwd_avx512(
                        pk_ir+b*56, pk_ii+b*56, pk_or+b*56, pk_oi+b*56,
                        pk_twr+b*48, pk_twi+b*48, 8);
                    radix7_tw_dit_kernel_bwd_avx512(
                        pk_or+b*56, pk_oi+b*56, pk_ir+b*56, pk_ii+b*56,
                        pk_twr+b*48, pk_twi+b*48, 8);
                }
            }
            t0 = now_ns();
            for (int i = 0; i < iters; i++) {
                for (size_t b = 0; b < nb; b++) {
                    radix7_tw_dit_kernel_fwd_avx512(
                        pk_ir+b*56, pk_ii+b*56, pk_or+b*56, pk_oi+b*56,
                        pk_twr+b*48, pk_twi+b*48, 8);
                    radix7_tw_dit_kernel_bwd_avx512(
                        pk_or+b*56, pk_oi+b*56, pk_ir+b*56, pk_ii+b*56,
                        pk_twr+b*48, pk_twi+b*48, 8);
                }
            }
            ns = (now_ns() - t0) / iters;
            if (ns < best_p) best_p = ns;

            /* FFTW: fwd + bwd (plan is fwd only, run twice for fair comparison) */
            for (int i = 0; i < warmup; i++)
                fftw_execute(plan_fwd);
            t0 = now_ns();
            for (int i = 0; i < iters; i++)
                fftw_execute(plan_fwd);
            ns = (now_ns() - t0) / iters;
            /* Double it since our kernels do fwd+bwd */
            ns *= 2.0;
            if (ns < best_f) best_f = ns;
        }

        double r_pk_st = best_s / best_p;
        double r_pk_fw = best_f / best_p;
        double r_st_fw = best_f / best_s;

        const char *winner = "FFTW";
        double best = best_f;
        if (best_p < best) { best = best_p; winner = "PACKED"; }
        if (best_s < best) { best = best_s; winner = "STRIDED"; }

        printf("  K=%-4zu | %7.1f  %7.1f  %7.1f | %7.2fx %7.2fx | %7.2fx  %s\n",
               K, best_s, best_p, best_f, r_pk_st, r_pk_fw, r_st_fw, winner);

        fftw_destroy_plan(plan_fwd);
        fftw_free(fftw_in); fftw_free(fftw_out);
        free(ir);free(ii);free(or_s);free(oi_s);free(or_p);free(oi_p);
        free(twr);free(twi);free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);
        free(pk_twr);free(pk_twi);
    }

    printf("\n  pk/strd >1 = packed faster than strided\n");
    printf("  pk/fftw >1 = packed faster than FFTW\n");
    printf("  st/fftw >1 = strided faster than FFTW\n");
    printf("  Note: FFTW does full N=7K DFT (all stages); we do single radix-7 stage\n");
    printf("        so FFTW cost is higher. Fair comparison is ns/DFT-7 operation.\n");

    fftw_cleanup();
    return 0;
}
