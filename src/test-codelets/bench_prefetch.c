/**
 * bench_prefetch.c -- Isolated t1 codelet comparison: normal vs prefetch
 *
 * Fair A/B: same full butterfly, one version has _mm_prefetch for
 * next block's twiddles. Tests R=8, R=64 at increasing M.
 * Also tests R=64 flat vs log3.
 *
 * No FFTW calls — pure isolated codelet timing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bench_compat.h"

#include "fft_radix8_avx2.h"
#include "r64_unified_avx2.h"

typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);

static void init_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* Bench one t1 variant, return ns */
static double bench_t1(t1_fn fn, double *rio_re, double *rio_im,
                       const double *W_re, const double *W_im,
                       size_t ios, size_t me, int reps)
{
    for (int i = 0; i < 20; i++) fn(rio_re, rio_im, W_re, W_im, ios, me);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fn(rio_re, rio_im, W_re, W_im, ios, me);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

static void test_pair(size_t R, size_t M, const char *name_a, t1_fn fn_a,
                      const char *name_b, t1_fn fn_b)
{
    size_t N = R * M;
    int reps = (int)(2e6 / (N+1));
    if (reps < 200) reps = 200;
    if (reps > 200000) reps = 200000;

    double *rio_re = (double*)aligned_alloc(32, N*8);
    double *rio_im = (double*)aligned_alloc(32, N*8);
    double *W_re = (double*)aligned_alloc(32, (R-1)*M*8);
    double *W_im = (double*)aligned_alloc(32, (R-1)*M*8);
    srand(42);
    for (size_t i = 0; i < N; i++) {
        rio_re[i] = (double)rand()/RAND_MAX - 0.5;
        rio_im[i] = (double)rand()/RAND_MAX - 0.5;
    }
    init_tw(W_re, W_im, R, M);
    size_t tw_kb = (R-1) * M * 16 / 1024;

    double ns_a = bench_t1(fn_a, rio_re, rio_im, W_re, W_im, M, M, reps);
    double ns_b = bench_t1(fn_b, rio_re, rio_im, W_re, W_im, M, M, reps);

    const char *winner = ns_b < ns_a ? name_b : name_a;
    double speedup = ns_a / ns_b;
    printf("    R=%-2zu M=%-4zu tw=%3zuKB  %s=%6.0f  %s=%6.0f  (%.2fx) %s\n",
           R, M, tw_kb, name_a, ns_a, name_b, ns_b, speedup,
           speedup > 1.02 ? "<-- B wins" : speedup < 0.98 ? "<-- A wins" : "~same");

    aligned_free(rio_re); aligned_free(rio_im);
    aligned_free(W_re); aligned_free(W_im);
}

int main(void) {
    printf("================================================================\n");
    printf("  Isolated t1 Codelet Comparison\n");
    printf("  L1=48KB. Same butterfly, different twiddle strategies.\n");
    printf("================================================================\n\n");
    fflush(stdout);

    /* Part 1: R=8 normal vs prefetch */
    printf("  R=8: t1_dit vs t1_dit_prefetch\n\n");
    size_t Ms_r8[] = {16, 32, 64, 128, 256, 512, 1024, 0};
    for (size_t *p = Ms_r8; *p; p++) {
        test_pair(8, *p, "normal", (t1_fn)radix8_t1_dit_fwd_avx2,
                        "prefetch", (t1_fn)radix8_t1_dit_prefetch_fwd_avx2);
        fflush(stdout);
    }

    /* Part 2: R=64 flat vs prefetch */
    printf("\n  R=64: t1_dit (flat) vs t1_dit_prefetch\n\n");
    size_t Ms_r64[] = {4, 8, 16, 32, 64, 128, 256, 0};
    for (size_t *p = Ms_r64; *p; p++) {
        test_pair(64, *p, "flat", (t1_fn)radix64_t1_dit_fwd_avx2,
                         "prefetch", (t1_fn)radix64_t1_dit_prefetch_fwd_avx2);
        fflush(stdout);
    }

    /* Part 3: R=64 flat vs log3 */
    printf("\n  R=64: t1_dit (flat) vs t1_dit_log3\n\n");
    for (size_t *p = Ms_r64; *p; p++) {
        test_pair(64, *p, "flat", (t1_fn)radix64_t1_dit_fwd_avx2,
                         "log3", (t1_fn)radix64_t1_dit_log3_fwd_avx2);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
