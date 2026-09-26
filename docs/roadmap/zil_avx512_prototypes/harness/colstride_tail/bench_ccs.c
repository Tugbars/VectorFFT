/* bench_ccs: n1ccs (lane k = whole transform at pitch Gs) avx2 original vs fixed avx512 twin,
 * in-place, count transforms of length RADIX, Gs = RADIX, Ls = 1. ns per call. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void SYM_A(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void SYM_B(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
typedef void (*zfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double run(zfn f, double *buf, size_t count)
{
    double best = 1e30;
    for (int t = 0; t < 21; t++) {
        double t0 = now();
        for (int r = 0; r < 500; r++) { f(buf, 0, buf, 0, 0, 0, 1, RADIX, 1, RADIX, count); __asm__ volatile("" ::: "memory"); }
        double dt = (now() - t0) / 500; if (dt < best) best = dt;
    }
    return best;
}
int main(void)
{
    size_t counts[] = { 8, 64, 63 };
    double *buf = aligned_alloc(64, sizeof(double) * 2 * RADIX * 64 + 64);
    for (size_t i = 0; i < 2 * RADIX * 64; i++) buf[i] = 1e-3 * (double)(i % 7);
    printf("%-22s", STRN);
    for (int c = 0; c < 3; c++) printf("  count=%zu: avx2 %.1f / avx512 %.1f", counts[c], run((zfn)SYM_A, buf, counts[c]), run((zfn)SYM_B, buf, counts[c]));
    printf("  ns/call\n");
    return 0;
}
