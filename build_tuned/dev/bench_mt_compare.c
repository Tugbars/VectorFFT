/* bench_mt_compare.c — compare MT scaling: 1D C2C (direct MT) vs DCT-II
 * (wrapper MT) at the same (N, K). If C2C scales similarly, the bottleneck
 * is hardware (memory bandwidth, hybrid cores), not our wrapper MT logic.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(void) {
    vfft_init();
    vfft_pin_thread(0);

    int Ns[] = { 256, 1024, 4096 };
    size_t Ks[] = { 1024, 4096 };
    int Ts[] = { 1, 2, 4, 8 };

    printf("=== bench_mt_compare -- 1D C2C vs DCT-II at same (N, K) ===\n\n");
    printf("%-6s %-7s %-7s  T=1            T=2            T=4            T=8\n",
           "kind", "N", "K");
    printf("------------------------------------------------------------------\n");

    for (int ni = 0; ni < 3; ni++) for (int ki = 0; ki < 2; ki++) {
        int N = Ns[ni]; size_t K = Ks[ki];
        size_t NK = (size_t)N * K;
        double *re = (double *)vfft_alloc(NK * sizeof(double));
        double *im = (double *)vfft_alloc(NK * sizeof(double));
        double *out = (double *)vfft_alloc(NK * sizeof(double));
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) {
            re[i] = (double)rand()/RAND_MAX - 0.5;
            im[i] = (double)rand()/RAND_MAX - 0.5;
        }

        /* C2C scaling */
        double c2c_ns[4];
        for (int ti = 0; ti < 4; ti++) {
            vfft_set_num_threads(Ts[ti]);
            vfft_plan p = vfft_plan_c2c(N, K, VFFT_ESTIMATE);
            for (int i = 0; i < 5; i++) vfft_execute_fwd(p, re, im);
            double best = 1e30;
            for (int i = 0; i < 21; i++) {
                double t0 = now_ns();
                vfft_execute_fwd(p, re, im);
                double dt = now_ns() - t0;
                if (dt < best) best = dt;
            }
            c2c_ns[ti] = best;
            vfft_destroy(p);
        }

        /* DCT-II scaling */
        double dct_ns[4];
        for (int ti = 0; ti < 4; ti++) {
            vfft_set_num_threads(Ts[ti]);
            vfft_plan p = vfft_plan_dct2(N, K, VFFT_ESTIMATE);
            for (int i = 0; i < 5; i++) vfft_execute_dct2(p, re, out);
            double best = 1e30;
            for (int i = 0; i < 21; i++) {
                double t0 = now_ns();
                vfft_execute_dct2(p, re, out);
                double dt = now_ns() - t0;
                if (dt < best) best = dt;
            }
            dct_ns[ti] = best;
            vfft_destroy(p);
        }

        printf("C2C    N=%-5d K=%-5zu  %8.0f       %8.0f (%.2fx)  %8.0f (%.2fx)  %8.0f (%.2fx)\n",
               N, K, c2c_ns[0],
               c2c_ns[1], c2c_ns[0]/c2c_ns[1],
               c2c_ns[2], c2c_ns[0]/c2c_ns[2],
               c2c_ns[3], c2c_ns[0]/c2c_ns[3]);
        printf("DCT-II N=%-5d K=%-5zu  %8.0f       %8.0f (%.2fx)  %8.0f (%.2fx)  %8.0f (%.2fx)\n",
               N, K, dct_ns[0],
               dct_ns[1], dct_ns[0]/dct_ns[1],
               dct_ns[2], dct_ns[0]/dct_ns[2],
               dct_ns[3], dct_ns[0]/dct_ns[3]);
        printf("\n");

        vfft_free(re); vfft_free(im); vfft_free(out);
    }

    vfft_set_num_threads(1);
    return 0;
}
