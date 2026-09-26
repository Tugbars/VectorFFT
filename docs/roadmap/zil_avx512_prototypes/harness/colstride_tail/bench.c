/* bench: narrow (xmm ladder, SYM_N) vs masked (zmm k-mask, SYM_M) tail, same kernel.
 * -DRADIX -DKIND(0/1 = T2 needs table, 2 = N1) ; ns per call, min of trials. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
extern void SYM_N(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void SYM_M(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
typedef void (*zfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double timeit(zfn f, const double *in, double *out, const double *tw, size_t Ls, size_t count, int reps)
{
    double best = 1e30;
    for (int t = 0; t < 15; t++) {
        double t0 = now();
        for (int r = 0; r < reps; r++) {
            f(in, 0, out, 0, tw, 0, Ls, 0, Ls, 0, count);
            __asm__ volatile("" ::: "memory");
        }
        double dt = (now() - t0) / reps;
        if (dt < best) best = dt;
    }
    return best;
}
int main(void)
{
    static const int counts[] = { 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 61, 62, 63 };
    size_t maxc = 64, Ls = maxc + 4;
    double *in = aligned_alloc(64, sizeof(double) * 2 * RADIX * Ls + 4096);
    double *out = aligned_alloc(64, sizeof(double) * 2 * RADIX * Ls + 4096);
    double *tw = aligned_alloc(64, sizeof(double) * 16 * RADIX * (maxc / 4 + 1) + 4096);
    for (size_t i = 0; i < 2 * RADIX * Ls; i++) in[i] = (double)(i % 97) / 97.0 - 0.5;
    for (size_t i = 0; i < 16 * RADIX * (maxc / 4 + 1); i++) tw[i] = 0.7;
    printf("%-28s", STRN);
    for (unsigned c = 0; c < sizeof counts / sizeof counts[0]; c++) {
        size_t n = (size_t)counts[c];
        int reps = (int)(200000 / (RADIX * (n + 3)));
        if (reps < 200) reps = 200;
        double tn = timeit((zfn)SYM_N, in, out, tw, Ls, n, reps);
        double tm = timeit((zfn)SYM_M, in, out, tw, Ls, n, reps);
        printf(" c%zu:%.1f/%.1f", n, tn, tm);
    }
    printf("   (ns/call narrow/masked)\n");
    return 0;
}
