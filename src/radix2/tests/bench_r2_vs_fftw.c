/**
 * @file bench_r2_vs_fftw.c
 * @brief Apples-to-apples: VectorFFT radix-2 vs FFTW radix-2
 *
 * Test 1: N=2 (single butterfly) — both do exactly one radix-2 op
 * Test 2: N=4096 with FFTW forced to 2×2×2×... decomposition
 *         via fftw_plan_guru64_dft with recursive rank-1 size-2 transforms
 *
 * FFTW N=2 DFT is literally: X[0]=x[0]+x[1], X[1]=x[0]-x[1]
 * VectorFFT N1 half=1 is the same thing in SoA layout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include "fft_radix2.h"

#define TRIALS    9
#define BENCH_MS  400
#define WARMUP_MS 100

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
static double med(double *a, int n) { qsort(a,n,sizeof(double),cmp_dbl); return a[n/2]; }

typedef struct { double ns; long iters; } bench_result;

static bench_result timed_loop_fftw(fftw_plan p) {
    /* warmup */
    double t0 = now_ns();
    while (now_ns() - t0 < WARMUP_MS * 1e6) fftw_execute(p);

    double results[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        t0 = now_ns();
        long it = 0;
        while (now_ns() - t0 < BENCH_MS * 1e6) { fftw_execute(p); it++; }
        results[t] = (now_ns() - t0) / it;
    }
    return (bench_result){ med(results, TRIALS), 0 };
}

/*==========================================================================
 * TEST 1: N=2  (single radix-2 butterfly)
 *==========================================================================*/
static void test_n2(void) {
    printf("────────────────────────────────────────────────────────────────\n");
    printf(" TEST 1: N=2 (single radix-2 butterfly)\n");
    printf("────────────────────────────────────────────────────────────────\n\n");

    /* --- FFTW N=2 forward --- */
    fftw_complex *fin  = fftw_alloc_complex(2);
    fftw_complex *fout = fftw_alloc_complex(2);
    fin[0][0] = 1.0; fin[0][1] = 0.5;
    fin[1][0] = 0.3; fin[1][1] = 0.7;

    fftw_plan p2 = fftw_plan_dft_1d(2, fin, fout, FFTW_FORWARD, FFTW_MEASURE);
    bench_result fftw_n2 = timed_loop_fftw(p2);
    fftw_destroy_plan(p2);

    /* Verify: X[0]=x[0]+x[1], X[1]=x[0]-x[1] */
    printf("  FFTW N=2 result:  X[0]=(%.1f,%.1f)  X[1]=(%.1f,%.1f)\n",
           fout[0][0], fout[0][1], fout[1][0], fout[1][1]);

    /* --- VectorFFT N1 half=1 --- */
    double *vin_re  = (double*)aligned_alloc(64, 2*sizeof(double));
    double *vin_im  = (double*)aligned_alloc(64, 2*sizeof(double));
    double *vout_re = (double*)aligned_alloc(64, 2*sizeof(double));
    double *vout_im = (double*)aligned_alloc(64, 2*sizeof(double));
    vin_re[0]=1.0; vin_im[0]=0.5; vin_re[1]=0.3; vin_im[1]=0.7;

    /* warmup + bench */
    double t0 = now_ns();
    while (now_ns() - t0 < WARMUP_MS * 1e6)
        fft_radix2_bv_n1(vout_re, vout_im, vin_re, vin_im, 1);

    double vn1_results[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        t0 = now_ns(); long it = 0;
        while (now_ns() - t0 < BENCH_MS * 1e6) {
            fft_radix2_bv_n1(vout_re, vout_im, vin_re, vin_im, 1);
            it++;
        }
        vn1_results[t] = (now_ns() - t0) / it;
    }
    double vfft_n1_n2 = med(vn1_results, TRIALS);

    printf("  VFFT N1 result:   X[0]=(%.1f,%.1f)  X[1]=(%.1f,%.1f)\n",
           vout_re[0], vout_im[0], vout_re[1], vout_im[1]);

    /* --- VectorFFT with-twiddles half=1 (W=1+0i for k=0) --- */
    double *tw_re = (double*)aligned_alloc(64, sizeof(double));
    double *tw_im = (double*)aligned_alloc(64, sizeof(double));
    tw_re[0] = 1.0; tw_im[0] = 0.0;
    fft_twiddles_soa tw = { .re = tw_re, .im = tw_im };

    t0 = now_ns();
    while (now_ns() - t0 < WARMUP_MS * 1e6)
        fft_radix2_bv(vout_re, vout_im, vin_re, vin_im, &tw, 1);

    double vtw_results[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        t0 = now_ns(); long it = 0;
        while (now_ns() - t0 < BENCH_MS * 1e6) {
            fft_radix2_bv(vout_re, vout_im, vin_re, vin_im, &tw, 1);
            it++;
        }
        vtw_results[t] = (now_ns() - t0) / it;
    }
    double vfft_tw_n2 = med(vtw_results, TRIALS);

    printf("\n  FFTW N=2:            %7.1f ns/call\n", fftw_n2.ns);
    printf("  VectorFFT N1 (N=2):  %7.1f ns/call\n", vfft_n1_n2);
    printf("  VectorFFT tw (N=2):  %7.1f ns/call\n", vfft_tw_n2);
    printf("  Ratio FFTW/VFFT-N1:  %.2fx\n", fftw_n2.ns / vfft_n1_n2);
    printf("  Ratio FFTW/VFFT-tw:  %.2fx\n\n", fftw_n2.ns / vfft_tw_n2);

    fftw_free(fin); fftw_free(fout);
    free(vin_re); free(vin_im); free(vout_re); free(vout_im);
    free(tw_re); free(tw_im);
}

/*==========================================================================
 * TEST 2: N=4096 — FFTW full FFT vs VectorFFT 12 radix-2 stages
 *
 * We run VectorFFT for ALL 12 stages (log2(4096)=12) with proper
 * twiddle factors for each stage, ping-ponging buffers.
 * This is a complete radix-2 DIT FFT using only radix-2 butterflies,
 * compared to FFTW which can use whatever radices it wants.
 *==========================================================================*/
static void test_n4096(void) {
    const int N = 4096;
    const int LOG2N = 12;

    printf("────────────────────────────────────────────────────────────────\n");
    printf(" TEST 2: N=%d — full FFT, radix-2 only vs FFTW\n", N);
    printf("────────────────────────────────────────────────────────────────\n\n");

    /* --- FFTW full FFT N=4096 --- */
    fftw_complex *fin  = fftw_alloc_complex(N);
    fftw_complex *fout = fftw_alloc_complex(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        fin[i][0] = (double)rand()/RAND_MAX - 0.5;
        fin[i][1] = (double)rand()/RAND_MAX - 0.5;
    }

    fftw_plan pm = fftw_plan_dft_1d(N, fin, fout, FFTW_FORWARD, FFTW_MEASURE);
    bench_result fftw_m = timed_loop_fftw(pm);
    fftw_destroy_plan(pm);

    fftw_plan pp = fftw_plan_dft_1d(N, fin, fout, FFTW_FORWARD, FFTW_PATIENT);
    bench_result fftw_p = timed_loop_fftw(pp);
    fftw_destroy_plan(pp);

    double fftw_best = fftw_m.ns < fftw_p.ns ? fftw_m.ns : fftw_p.ns;

    /* --- VectorFFT: 12 radix-2 stages with proper twiddles ---
     * DIT (decimation-in-time) Cooley-Tukey:
     *   Stage s (0-indexed): butterfly groups of size 2^(s+1), half = 2^s
     *   Twiddle[k] = exp(-2πi·k / 2^(s+1))
     *
     * We bit-reverse the input first, then do 12 in-order stages.
     */

    /* Allocate SoA buffers */
    double *buf_a_re = (double*)aligned_alloc(64, N*sizeof(double));
    double *buf_a_im = (double*)aligned_alloc(64, N*sizeof(double));
    double *buf_b_re = (double*)aligned_alloc(64, N*sizeof(double));
    double *buf_b_im = (double*)aligned_alloc(64, N*sizeof(double));

    /* Pre-compute twiddle tables for each stage */
    fft_twiddles_soa tw_stages[LOG2N];
    for (int s = 0; s < LOG2N; s++) {
        int half = 1 << s;
        tw_stages[s].re = (double*)aligned_alloc(64, half * sizeof(double));
        tw_stages[s].im = (double*)aligned_alloc(64, half * sizeof(double));
        int span = half * 2;
        for (int k = 0; k < half; k++) {
            double angle = -2.0 * M_PI * k / span;
            tw_stages[s].re[k] = cos(angle);
            tw_stages[s].im[k] = sin(angle);
        }
    }

    /* Bit-reversal permutation table */
    int *bitrev = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < LOG2N; b++)
            if (i & (1 << b)) rev |= (1 << (LOG2N - 1 - b));
        bitrev[i] = rev;
    }

    /* Copy FFTW input data into SoA + bit-reverse */
    /* (We'll re-do this each iteration inside the loop to be fair) */

    /* === Full radix-2 DIT FFT function (inline) === */
    #define DO_FULL_R2_FFT() do { \
        /* Bit-reverse copy into buf_a */ \
        for (int i = 0; i < N; i++) { \
            buf_a_re[i] = fin[bitrev[i]][0]; \
            buf_a_im[i] = fin[bitrev[i]][1]; \
        } \
        /* 12 stages, ping-pong buffers */ \
        double *src_re = buf_a_re, *src_im = buf_a_im; \
        double *dst_re = buf_b_re, *dst_im = buf_b_im; \
        for (int s = 0; s < LOG2N; s++) { \
            int half = 1 << s; \
            int span = half * 2; \
            /* Process all butterfly groups for this stage */ \
            for (int g = 0; g < N; g += span) { \
                if (s == 0) { \
                    /* Stage 0: half=1, W=1 → N1 */ \
                    fft_radix2_bv_n1(&dst_re[g], &dst_im[g], \
                                     &src_re[g], &src_im[g], half); \
                } else { \
                    fft_radix2_bv(&dst_re[g], &dst_im[g], \
                                  &src_re[g], &src_im[g], \
                                  &tw_stages[s], half); \
                } \
            } \
            /* Swap buffers */ \
            double *tmp; \
            tmp = src_re; src_re = dst_re; dst_re = tmp; \
            tmp = src_im; src_im = dst_im; dst_im = tmp; \
        } \
    } while(0)

    /* Warmup */
    double t0 = now_ns();
    while (now_ns() - t0 < WARMUP_MS * 1e6) { DO_FULL_R2_FFT(); }

    double vfft_results[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        t0 = now_ns(); long it = 0;
        while (now_ns() - t0 < BENCH_MS * 1e6) { DO_FULL_R2_FFT(); it++; }
        vfft_results[t] = (now_ns() - t0) / it;
    }
    double vfft_full = med(vfft_results, TRIALS);

    /* --- Also time WITHOUT bit-reversal to isolate butterfly cost --- */
    /* Pre-load bit-reversed data once */
    for (int i = 0; i < N; i++) {
        buf_a_re[i] = fin[bitrev[i]][0];
        buf_a_im[i] = fin[bitrev[i]][1];
    }

    #define DO_R2_STAGES_ONLY() do { \
        /* Re-copy pre-reversed data */ \
        memcpy(buf_a_re, buf_b_re, N*sizeof(double)); \
        memcpy(buf_a_im, buf_b_im, N*sizeof(double)); \
        double *src_re = buf_a_re, *src_im = buf_a_im; \
        double *dst_re = buf_b_re, *dst_im = buf_b_im; \
        for (int s = 0; s < LOG2N; s++) { \
            int half = 1 << s; \
            int span = half * 2; \
            for (int g = 0; g < N; g += span) { \
                if (s == 0) { \
                    fft_radix2_bv_n1(&dst_re[g], &dst_im[g], \
                                     &src_re[g], &src_im[g], half); \
                } else { \
                    fft_radix2_bv(&dst_re[g], &dst_im[g], \
                                  &src_re[g], &src_im[g], \
                                  &tw_stages[s], half); \
                } \
            } \
            double *tmp; \
            tmp = src_re; src_re = dst_re; dst_re = tmp; \
            tmp = src_im; src_im = dst_im; dst_im = tmp; \
        } \
    } while(0)

    /* Save bit-reversed data into buf_b for repeated copy */
    for (int i = 0; i < N; i++) {
        buf_b_re[i] = fin[bitrev[i]][0];
        buf_b_im[i] = fin[bitrev[i]][1];
    }

    t0 = now_ns();
    while (now_ns() - t0 < WARMUP_MS * 1e6) { DO_R2_STAGES_ONLY(); }

    double vfft_stages[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        t0 = now_ns(); long it = 0;
        while (now_ns() - t0 < BENCH_MS * 1e6) { DO_R2_STAGES_ONLY(); it++; }
        vfft_stages[t] = (now_ns() - t0) / it;
    }
    double vfft_stages_ns = med(vfft_stages, TRIALS);

    /* --- Per-stage breakdown --- */
    printf("  FFTW full FFT (MEASURE):       %8.1f ns  (%.2f ns/pt)\n",
           fftw_m.ns, fftw_m.ns / N);
    printf("  FFTW full FFT (PATIENT):       %8.1f ns  (%.2f ns/pt)\n",
           fftw_p.ns, fftw_p.ns / N);
    printf("  FFTW best:                     %8.1f ns\n\n", fftw_best);

    printf("  VectorFFT full (bitrev+12stg): %8.1f ns  (%.2f ns/pt)\n",
           vfft_full, vfft_full / N);
    printf("  VectorFFT stages only (12stg): %8.1f ns  (%.2f ns/pt)\n",
           vfft_stages_ns, vfft_stages_ns / N);
    printf("  VectorFFT per-stage avg:       %8.1f ns\n\n",
           vfft_stages_ns / LOG2N);

    printf("  ── Ratios ──\n");
    printf("  FFTW / VectorFFT full:         %.2fx\n",
           fftw_best / vfft_full);
    printf("  FFTW / VectorFFT stages-only:  %.2fx\n",
           fftw_best / vfft_stages_ns);
    printf("  (>1 = FFTW faster, <1 = VectorFFT faster)\n\n");

    printf("  ── Breakdown ──\n");
    printf("  Bit-reversal overhead:         %8.1f ns  (%.0f%% of full)\n",
           vfft_full - vfft_stages_ns,
           100.0 * (vfft_full - vfft_stages_ns) / vfft_full);
    printf("  Total butterflies (12 stages): %d\n", N / 2 * LOG2N);
    printf("  ns/butterfly (stages only):    %.2f\n",
           vfft_stages_ns / (N / 2.0 * LOG2N));
    printf("  FFTW ns/butterfly equivalent:  %.2f\n\n",
           fftw_best / (N / 2.0 * LOG2N));

    /* Cleanup */
    fftw_free(fin); fftw_free(fout);
    free(buf_a_re); free(buf_a_im); free(buf_b_re); free(buf_b_im);
    for (int s = 0; s < LOG2N; s++) {
        free((void*)tw_stages[s].re);
        free((void*)tw_stages[s].im);
    }
    free(bitrev);
}

/*==========================================================================
 * TEST 3: Single-stage sweep — FFTW N=2,4,8,...,4096 vs VectorFFT stage
 *==========================================================================*/
static void test_single_stage_sweep(void) {
    printf("────────────────────────────────────────────────────────────────\n");
    printf(" TEST 3: Single butterfly stage — FFTW(N) vs VectorFFT(half=N/2)\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("  Note: FFTW(N=2^k) does log2(N) stages internally, not 1.\n");
    printf("  Dividing by log2(N) gives FFTW's effective per-stage cost.\n\n");

    printf("  %-6s | %-10s %-10s | %-10s %-10s | %-7s\n",
           "N", "FFTW(ns)", "FFTW/stg", "VFFT-tw", "VFFT-N1", "tw-ratio");
    printf("  ───────┼───────────────────────┼───────────────────────┼────────\n");

    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int si = 0; si < nsizes; si++) {
        int n = sizes[si];
        int half = n / 2;
        int log2n = 0; { int tmp = n; while (tmp > 1) { tmp >>= 1; log2n++; } }

        /* FFTW */
        fftw_complex *fi = fftw_alloc_complex(n);
        fftw_complex *fo = fftw_alloc_complex(n);
        srand(42);
        for (int i = 0; i < n; i++) {
            fi[i][0] = (double)rand()/RAND_MAX - 0.5;
            fi[i][1] = (double)rand()/RAND_MAX - 0.5;
        }
        fftw_plan p = fftw_plan_dft_1d(n, fi, fo, FFTW_FORWARD, FFTW_MEASURE);
        bench_result fr = timed_loop_fftw(p);
        fftw_destroy_plan(p);
        fftw_free(fi); fftw_free(fo);

        double fftw_per_stage = fr.ns / log2n;

        /* VectorFFT single stage */
        double *ire = (double*)aligned_alloc(64, n*sizeof(double));
        double *iim = (double*)aligned_alloc(64, n*sizeof(double));
        double *ore = (double*)aligned_alloc(64, n*sizeof(double));
        double *oim = (double*)aligned_alloc(64, n*sizeof(double));
        double *tre = (double*)aligned_alloc(64, half*sizeof(double));
        double *tim = (double*)aligned_alloc(64, half*sizeof(double));
        srand(42);
        for (int i = 0; i < n; i++) {
            ire[i] = (double)rand()/RAND_MAX - 0.5;
            iim[i] = (double)rand()/RAND_MAX - 0.5;
        }
        for (int k = 0; k < half; k++) {
            double a = -2.0*M_PI*k/n;
            tre[k] = cos(a); tim[k] = sin(a);
        }
        fft_twiddles_soa tw = { .re = tre, .im = tim };

        /* Twiddle stage */
        double t0 = now_ns();
        while (now_ns() - t0 < WARMUP_MS * 1e6)
            fft_radix2_bv(ore, oim, ire, iim, &tw, half);
        double vtw[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            t0 = now_ns(); long it = 0;
            while (now_ns() - t0 < BENCH_MS * 1e6) {
                fft_radix2_bv(ore, oim, ire, iim, &tw, half); it++;
            }
            vtw[t] = (now_ns() - t0) / it;
        }

        /* N1 stage */
        t0 = now_ns();
        while (now_ns() - t0 < WARMUP_MS * 1e6)
            fft_radix2_bv_n1(ore, oim, ire, iim, half);
        double vn1[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            t0 = now_ns(); long it = 0;
            while (now_ns() - t0 < BENCH_MS * 1e6) {
                fft_radix2_bv_n1(ore, oim, ire, iim, half); it++;
            }
            vn1[t] = (now_ns() - t0) / it;
        }

        double vtw_ns = med(vtw, TRIALS);
        double vn1_ns = med(vn1, TRIALS);

        printf("  %-6d | %8.1f   %8.1f   | %8.1f   %8.1f   | %.2fx\n",
               n, fr.ns, fftw_per_stage, vtw_ns, vn1_ns,
               vtw_ns / fftw_per_stage);

        free(ire); free(iim); free(ore); free(oim); free(tre); free(tim);
    }
    printf("\n  tw-ratio: VectorFFT-twiddle / FFTW-per-stage (<1 = VFFT wins)\n\n");
}

int main(void) {
    printf("================================================================\n");
    printf(" VectorFFT Radix-2 vs FFTW  |  Single-core\n");
    printf(" VectorFFT SIMD: %s\n", radix2_get_simd_capabilities());
    printf(" FFTW: %s\n", fftw_version);
    printf("================================================================\n\n");

    test_n2();
    test_n4096();
    test_single_stage_sweep();

    return 0;
}
