/* bench_vfft_mt.c — VectorFFT T=8 latency sweep for cuFFT comparison.
 *
 * Same N × K grid as bench_vs_cufft.cu so results align cell-for-cell.
 *
 * Wisdom is loaded if available (K=8 is not in the v1.0 wisdom grid, so
 * those plans use VFFT_ESTIMATE — but K=64 and K=256 hit wisdom on N
 * values that are calibrated). The strategic question is "what does
 * VectorFFT do at T=8 for various K and N", which means we should run
 * with the planning mode users will actually use; ESTIMATE is the
 * default and the always-available path.
 *
 * Output: CSV columns N, K, vfft_t8_ns to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"

#define WARM 5
#define REPS 21

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(void) {
    /* ── Init + multithreading + thread pinning ────────────────────────
     * We set num_threads(8) BEFORE creating any plan. The plan captures
     * a "T_plan" snapshot at creation time and sizes scratch / dispatch
     * tables for that thread count. */
    vfft_init();
    vfft_pin_thread(0);                  /* pin caller to logical CPU 0 */
    vfft_set_num_threads(8);             /* T=8 — uses 8 P-cores on i9-14900KF */
    fprintf(stderr, "[vfft] num_threads=%d  ISA=%s  version=%s\n",
            vfft_get_num_threads(), vfft_isa(), vfft_version());

    /* Load packaged wisdom if available (T=8 plans pick up calibrated
     * factorizations for K∈{4,32,256}; K=8 will fall through to
     * estimate since wisdom only has K∈{4,32,256}). */
    int wrc = vfft_load_wisdom(
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[vfft] wisdom load rc=%d\n", wrc);

    int Ns[] = {64, 256, 1024, 4096, 16384, 65536, 262144};
    int Ks[] = {8, 128, 256};
    int n_Ns = sizeof(Ns) / sizeof(Ns[0]);
    int n_Ks = sizeof(Ks) / sizeof(Ks[0]);

    /* CSV header */
    printf("N,K,vfft_t8_ns\n");

    /* Allocate buffers for the largest cell. Reused across cells. */
    size_t max_NK = (size_t)Ns[n_Ns - 1] * Ks[n_Ks - 1];
    double *re = (double *)vfft_alloc(max_NK * sizeof(double));
    double *im = (double *)vfft_alloc(max_NK * sizeof(double));
    if (!re || !im) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    srand(42);
    for (size_t i = 0; i < max_NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    for (int ki = 0; ki < n_Ks; ki++) {
        int K = Ks[ki];
        for (int ni = 0; ni < n_Ns; ni++) {
            int N = Ns[ni];

            /* Use MEASURE for cells that have wisdom (K∈{4,32,256});
             * fall through to ESTIMATE for K=8. The wisdom-load above
             * populates g_wisdom for matching cells; if a cell isn't in
             * wisdom, MEASURE would calibrate it on-the-fly (slow).
             * For this bench we want fast plans, so use ESTIMATE — same
             * mode users get by default. */
            unsigned flags = VFFT_ESTIMATE;
            vfft_plan p = vfft_plan_c2c(N, (size_t)K, flags);
            if (!p) {
                fprintf(stderr, "plan_c2c(N=%d K=%d) failed\n", N, K);
                continue;
            }

            for (int w = 0; w < WARM; w++)
                vfft_execute_fwd(p, re, im);

            double best = 1e30;
            for (int r = 0; r < REPS; r++) {
                double t0 = now_ns();
                vfft_execute_fwd(p, re, im);
                double dt = now_ns() - t0;
                if (dt < best) best = dt;
            }

            printf("%d,%d,%.0f\n", N, K, best);
            fflush(stdout);

            vfft_destroy(p);
        }
    }

    vfft_free(re); vfft_free(im);
    return 0;
}
