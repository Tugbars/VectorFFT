/**
 * bench_radix5_flat.c — Compare R=5 flat vs log3 twiddle codelets
 *
 * Build: same includes as bench_full_fft
 *   add_executable(bench_radix5_flat bench_radix5_flat.c)
 *   target_compile_options(bench_radix5_flat PRIVATE -mavx2 -mfma -mno-avx512f)
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "vfft_common.h"     /* vfft_aligned_alloc/free */

/* Include BOTH codelets */
#include "fft_radix5_avx2.h"          /* existing log3 */
#include "fft_radix5_avx2_tw_flat.h"  /* new flat */

/* Timing */
#ifdef _WIN32
#include <windows.h>
static double get_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double get_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

static void bench_r5(size_t K)
{
    size_t N = 5 * K;
    double *ir  = (double*)vfft_aligned_alloc(64, N*8);
    double *ii  = (double*)vfft_aligned_alloc(64, N*8);
    double *or1 = (double*)vfft_aligned_alloc(64, N*8);
    double *oi1 = (double*)vfft_aligned_alloc(64, N*8);
    double *or2 = (double*)vfft_aligned_alloc(64, N*8);
    double *oi2 = (double*)vfft_aligned_alloc(64, N*8);

    size_t tsz = 4 * K;
    double *twr = (double*)vfft_aligned_alloc(64, tsz*8);
    double *twi = (double*)vfft_aligned_alloc(64, tsz*8);

    srand(42);
    for (size_t i = 0; i < N; i++) {
        ir[i] = (double)rand()/RAND_MAX*2-1;
        ii[i] = (double)rand()/RAND_MAX*2-1;
    }
    for (size_t n = 1; n < 5; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0*M_PI*(double)(n*k)/(double)N;
            twr[(n-1)*K+k] = cos(a);
            twi[(n-1)*K+k] = sin(a);
        }

    int reps = K <= 64 ? 20000 : K <= 512 ? 5000 : K <= 4096 ? 1000 : 200;

    /* Correctness check */
    radix5_tw_dit_kernel_fwd_avx2(ir,ii,or1,oi1,twr,twi,K);
    radix5_tw_flat_dit_kernel_fwd_avx2(ir,ii,or2,oi2,twr,twi,K);
    double maxerr = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(or1[i]-or2[i]) + fabs(oi1[i]-oi2[i]);
        if (e > maxerr) maxerr = e;
    }

    /* Bench log3 */
    for (int r = 0; r < 5; r++) radix5_tw_dit_kernel_fwd_avx2(ir,ii,or1,oi1,twr,twi,K);
    double best_log3 = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = get_ns();
        for (int r = 0; r < reps; r++) radix5_tw_dit_kernel_fwd_avx2(ir,ii,or1,oi1,twr,twi,K);
        double ns = (get_ns()-t0)/reps;
        if (ns < best_log3) best_log3 = ns;
    }

    /* Bench flat */
    for (int r = 0; r < 5; r++) radix5_tw_flat_dit_kernel_fwd_avx2(ir,ii,or2,oi2,twr,twi,K);
    double best_flat = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = get_ns();
        for (int r = 0; r < reps; r++) radix5_tw_flat_dit_kernel_fwd_avx2(ir,ii,or2,oi2,twr,twi,K);
        double ns = (get_ns()-t0)/reps;
        if (ns < best_flat) best_flat = ns;
    }

    /* Bench DIF log3 (current) */
    /* We don't have a DIF log3 flat equivalent yet, just compare DIT */

    double ratio = best_log3 / best_flat;
    printf("  K=%-6zu  log3=%7.0f  flat=%7.0f  ratio=%.2f  %s  err=%.1e\n",
           K, best_log3, best_flat, ratio,
           ratio > 1.05 ? "\033[92mFLAT WINS\033[0m" :
           ratio < 0.95 ? "log3 wins" : "~tie",
           maxerr);

    vfft_aligned_free(ir); vfft_aligned_free(ii);
    vfft_aligned_free(or1); vfft_aligned_free(oi1);
    vfft_aligned_free(or2); vfft_aligned_free(oi2);
    vfft_aligned_free(twr); vfft_aligned_free(twi);
}

int main(void)
{
    printf("═══════════════════════════════════════════════════\n");
    printf("  R=5 Twiddle Codelet: log3 vs flat (AVX2)\n");
    printf("═══════════════════════════════════════════════════\n\n");

    static const size_t Ks[] = {
        4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 0
    };

    printf("  %-8s  %9s  %9s  %7s  %s\n", "K", "log3_ns", "flat_ns", "ratio", "winner");
    printf("  %-8s  %9s  %9s  %7s  %s\n", "--------", "---------", "---------", "-------", "------");

    for (const size_t *kp = Ks; *kp; kp++)
        bench_r5(*kp);

    return 0;
}
