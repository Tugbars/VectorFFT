/* asan_c2r.c — minimal ASan harness to pin the c2r backward wild write.
 *
 * Stride (decoupled) c2r at increasing K; no MKL, no JIT. Linux aligned_alloc is
 * ASan-instrumented, so the library scratch + the bench buffers are all guarded.
 * ASan stops at the FIRST out-of-bounds write with the exact r2c.h file:line.
 *
 * Build (WSL): cd build_tuned && VFFT_ASAN=1 python3 build.py --src benches/asan_c2r.c --compile
 * Run  (WSL): ./benches/asan_c2r
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
#include "threads.h"
#include "env.h"
#include "planner.h"
#include "dp_planner.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include "generator/generated/registry.h"

int main(void)
{
    stride_env_init();
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    int Ns[] = {256, 512, 1024};
    size_t Ks[] = {8, 16, 32, 64, 128, 256};
    for (int ni = 0; ni < 3; ni++)
        for (int ki = 0; ki < 6; ki++)
        {
            int N = Ns[ni];
            size_t K = Ks[ki];
            stride_plan_t *inner = vfft_proto_auto_plan(N / 2, K, &reg, NULL);
            if (!inner) { printf("N=%d K=%zu inner NULL\n", N, K); continue; }
            stride_plan_t *sp = stride_r2c_plan(N, K, K, inner);
            if (!sp) { printf("N=%d K=%zu plan NULL\n", N, K); continue; }
            size_t total = (size_t)N * K, hcN = (size_t)(N / 2 + 1) * K;
            double *x = malloc(total * 8), *ore = malloc(hcN * 8), *oim = malloc(hcN * 8), *y = malloc(total * 8);
            srand(7);
            for (size_t i = 0; i < total; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
            stride_execute_r2c(sp, x, ore, oim);
            stride_execute_c2r(sp, ore, oim, y); /* ASan pins any OOB write here */
            double rel = 0;
            for (size_t i = 0; i < total; i++) { double e = fabs(y[i] - (double)N * x[i]); if (e > rel) rel = e; }
            printf("N=%-5d K=%-4zu rel=%.1e OK\n", N, K, rel);
            fflush(stdout);
            free(x); free(ore); free(oim); free(y);
            stride_plan_destroy(sp);
        }
    printf("ALL CELLS DONE (no ASan trap)\n");
    return 0;
}
