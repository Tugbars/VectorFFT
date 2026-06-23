/* asan_rfft_packed.c — WSL-ASan harness for the PACKED c2r path (rfft packed
 * forward + packed c2r), no MKL. Isolates whether the high-N*K crash is in the
 * rfft forward, the packed c2r, or (absent here) MKL. Linux aligned_alloc is
 * ASan-instrumented, so any OOB write traps at the exact r2c/rfft/c2r line.
 *
 * Build (WSL): cd build_tuned && VFFT_ASAN=1 python3 build.py --src benches/asan_rfft_packed.c --compile
 * Run  (WSL): ./benches/asan_rfft_packed
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "proto_stride_compat.h"
#include "env.h"
#include "rfft.h"
#include "c2r.h"
#include "rfft_registry_avx2.h"
#include "c2r_registry_avx2.h"
#include "c2r_dispatch.h"

int main(void)
{
    stride_env_init();
    rfft_codelets_t reg;
    memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);  /* fwd r2cf + hc2hc_dit */
    c2r_register_all_avx2(&reg);   /* bwd r2cb + hc2hc_dif */
    int Ns[] = {256, 512, 1024};
    size_t Ks[] = {8, 16, 32, 64, 128, 256};
    for (int ni = 0; ni < 3; ni++)
        for (int ki = 0; ki < 6; ki++)
        {
            int N = Ns[ni];
            size_t K = Ks[ki];
            c2r_plan_t *pb = vfft_c2r_plan_create(N, K, &reg);
            if (!pb) { printf("N=%-5d K=%-4zu plan NULL\n", N, K); continue; }
            size_t total = (size_t)N * K;
            double *x = malloc(total * 8), *hc = malloc(total * 2 * 8), *y = malloc(total * 8);
            srand(7);
            for (size_t i = 0; i < total; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(hc, 0, total * 2 * 8);
            rfft_execute_fwd_packed(pb->base, x, hc);  /* ASan: forward OOB? */
            c2r_execute_packed(pb, hc, y);             /* ASan: packed c2r OOB? */
            double rel = 0;
            for (size_t i = 0; i < total; i++) { double e = fabs(y[i] - (double)N * x[i]); if (e > rel) rel = e; }
            printf("N=%-5d K=%-4zu rel=%.1e OK\n", N, K, rel);
            fflush(stdout);
            free(x); free(hc); free(y);
            c2r_plan_destroy(pb);
        }
    printf("ALL PACKED CELLS DONE (no ASan trap)\n");
    return 0;
}
