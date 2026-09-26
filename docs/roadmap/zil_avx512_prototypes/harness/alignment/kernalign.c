/* kernalign.c — the alignment question on REAL generated IL codelets (INDICATIVE,
 * VM 4 vCPU). avx512 twin vs avx2 original of the same kernel, bulk only
 * (count % 4 == 0, so no tail), with zin/zout/tw base offsets 0/16/32/48 mod 64.
 * Ls = OLs = count, so every leg row keeps the base alignment. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#include "kdecls.h"   /* struct row { name; R; f; } rows[] */

static double now_ns(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

int main(void)
{
    size_t counts[] = { 64, 68, 1000, 1024 };
    char *rin = aligned_alloc(4096, 64u << 20), *rout = aligned_alloc(4096, 64u << 20),
         *rtw = aligned_alloc(4096, 64u << 20);
    memset(rin, 0, 64u << 20); memset(rout, 0, 64u << 20); memset(rtw, 0, 64u << 20);
    for (size_t i = 0; i < (64u << 20) / 8; i++) { ((double *)rin)[i] = (i % 13) * 0.1; ((double *)rtw)[i] = 0.5; }
    printf("ns per complex point per pass (R*count points), min of trials\n");
    printf("%-22s %6s %8s %8s %8s %8s   split-rate zmm/ymm at off 16|32|48\n", "kernel", "count", "off0", "off16", "off32", "off48");
    for (size_t r = 0; r < sizeof rows / sizeof rows[0]; r++)
        for (int c = 0; c < 4; c++) {
            size_t count = counts[c]; int R = rows[r].R;
            double res[4];
            for (int o = 0; o < 4; o++) {
                const double *in = (const double *)(rin + 16 * o);
                double *out = (double *)(rout + 16 * o);
                const double *tw = (const double *)(rtw + 16 * o);
                long reps = (long)(2e7 / (R * count)); if (reps < 5) reps = 5;
                double best = 1e30;
                for (int t = 0; t < 7; t++) {
                    double t0 = now_ns();
                    for (long i = 0; i < reps; i++) rows[r].f(in, 0, out, 0, tw, 0, count, 0, count, 0, count);
                    double dt = (now_ns() - t0) / reps / (R * count); if (dt < best) best = dt;
                }
                res[o] = best;
            }
            printf("%-22s %6zu %8.4f %8.4f %8.4f %8.4f\n", rows[r].name, count, res[0], res[1], res[2], res[3]);
        }
    return 0;
}
