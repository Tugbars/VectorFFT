/* bench_off: ns/call of one kernel (SYM) vs byte offset of the zin/zout base
 * from a 64-byte boundary. count = Ls = 64 (so legs stay at the same phase).
 * -DRADIX -DVW (4 or 8, table geometry) */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void SYM(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
int main(void)
{
    const size_t count = 64, Ls = 64;
    char *ib = aligned_alloc(64, 2 * RADIX * Ls * 8 + 256), *ob = aligned_alloc(64, 2 * RADIX * Ls * 8 + 256);
    double *tw = aligned_alloc(64, sizeof(double) * 2 * VW * RADIX * (count / (VW / 2) + 1));
    for (size_t i = 0; i < 2 * VW * RADIX * (count / (VW / 2) + 1); i++) tw[i] = 0.7;
    for (size_t i = 0; i < (2 * RADIX * Ls * 8 + 256) / 8; i++) ((double *)ib)[i] = (double)(i % 13) * 0.1;
    printf("%-26s", STRN);
    for (int off = 0; off < 64; off += 16) {
        const double *in = (const double *)(ib + off);
        double *out = (double *)(ob + off);
        double best = 1e30;
        for (int t = 0; t < 21; t++) {
            double t0 = now();
            for (int r = 0; r < 2000; r++) { SYM(in, 0, out, 0, tw, 0, Ls, 0, Ls, 0, count); __asm__ volatile("" ::: "memory"); }
            double dt = (now() - t0) / 2000;
            if (dt < best) best = dt;
        }
        printf("  off%2d: %7.1f", off, best);
    }
    printf("  ns/call (count=64)\n");
    return 0;
}
