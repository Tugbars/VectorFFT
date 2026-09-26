/* tailbench.c — INDICATIVE (VM, 4 vCPU, Emerald Rapids) cost of the IL remainder
 * at 4 complex/zmm under three tail policies, same kernel, same generator:
 *   masked    : ONE zmm pass, maskz loads / mask stores (_tm = (1<<2rem)-1)
 *   ladder    : ymm pass (2 complex) if rem>=2, then xmm pass if rem odd
 *   narrowfix : today's per-column xmm loop (rem iterations) + lane-offset fix
 * plus the avx2 kernel (2 complex/ymm, xmm odd tail) as a reference line.
 * Buffers are L1-resident; min over trials of the per-call mean. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#include "decls.h"   /* generated: extern decls + struct row rows[] */

static double now_ns(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

static double bench(kfn f, int R, size_t count, double *in, double *out, double *tw)
{
    size_t Ls = (count + 3) / 4 * 4 + LSPAD, OLs = Ls;
    int iters = count <= 4 ? 20000 : 8000;
    double best = 1e30;
    for (int w = 0; w < 200; w++) f(in, 0, out, 0, tw, 0, Ls, 0, OLs, 0, count);
    for (int t = 0; t < 15; t++) {
        double t0 = now_ns();
        for (int i = 0; i < iters; i++) {
            f(in, 0, out, 0, tw, 0, Ls, 0, OLs, 0, count);
            __asm__ volatile("" ::: "memory");
        }
        double dt = (now_ns() - t0) / iters;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv)
{
    size_t maxc = 64;
    double *in = aligned_alloc(64, 2 * 64 * maxc * sizeof(double) + 4096);
    double *out = aligned_alloc(64, 2 * 64 * maxc * sizeof(double) + 4096);
    double *tw = aligned_alloc(64, 64 * 64 * maxc * sizeof(double) + 4096);
    for (size_t i = 0; i < 2 * 64 * maxc; i++) in[i] = (double)(i % 97) * 0.01 - 0.3;
    for (size_t i = 0; i < 64 * 64 * maxc; i++) tw[i] = (double)(i % 89) * 0.011 - 0.4;
    size_t counts[] = { 1, 2, 3, 4, 8, 9, 10, 11, 12 };
    int nc = sizeof counts / sizeof counts[0];
    printf("%-10s %-10s", "kernel", "policy");
    for (int c = 0; c < nc; c++) printf(" %6zu", counts[c]);
    printf("   (ns/call; columns = count)\n");
    for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++) {
        printf("%-10s %-10s", rows[r].kern, rows[r].pol);
        for (int c = 0; c < nc; c++)
            printf(" %6.1f", bench(rows[r].f, rows[r].R, counts[c], in, out, tw));
        printf("\n");
    }
    return 0;
}
