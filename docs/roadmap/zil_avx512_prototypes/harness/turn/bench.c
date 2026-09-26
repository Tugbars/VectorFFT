/* bench.c -- n1t radix RAD: avx2 vs avx512 (narrow VEX-128 tail) vs avx512
 * (masked tail), L1-resident, as in il2p stage 1 (Ls = count, OLs = RAD).
 * Reports ns per call, median of 15 reps x (enough iterations). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#define S_(x) #x
#define S(x) S_(x)
#define CAT_(a,b) a##b
#define CAT(a,b) CAT_(a,b)
typedef void (*fn11)(const double *, const double *, double *, double *,
                     const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void CAT(CAT(radix, RAD), _z_n1t_fwd_avx2)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void CAT(CAT(radix, RAD), _z_n1t_fwd_avx512)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void CAT(CAT(radix, RAD), _z_n1t_fwd_avx512m)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static double timeit(fn11 f, const double *zin, double *zout, size_t count, int off)
{
    (void)off;
    const int R = RAD;
    double t[15];
    long iters = 2000000 / (long)(R * (count + 1)) + 50;
    for (int w = 0; w < 3; w++) f(zin, 0, zout, 0, 0, 0, count, 0, R, 0, count);
    for (int r = 0; r < 15; r++) {
        double t0 = now();
        for (long i = 0; i < iters; i++) { f(zin, 0, zout, 0, 0, 0, count, 0, R, 0, count); __asm__ volatile("" ::: "memory"); }
        t[r] = (now() - t0) / iters;
    }
    qsort(t, 15, sizeof(double), cmpd);
    return t[7];
}

int main(void)
{
    const int R = RAD;
    fn11 fs[3] = { CAT(CAT(radix, RAD), _z_n1t_fwd_avx2), CAT(CAT(radix, RAD), _z_n1t_fwd_avx512), CAT(CAT(radix, RAD), _z_n1t_fwd_avx512m) };
    const size_t counts[] = { 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 31, 32, 64 };
    const int offs[] = { 0, 16 };
    printf("radix %d n1t, ns/call (median of 15)\n", R);
    printf("%6s %4s %9s %9s %9s  %s\n", "count", "off", "avx2", "512narrow", "512mask", "mask/narrow");
    for (unsigned ci = 0; ci < sizeof counts / sizeof counts[0]; ci++)
    for (int oi = 0; oi < 2; oi++) {
        size_t count = counts[ci];
        double *ib = aligned_alloc(64, (2 * R * count + 64) * sizeof(double));
        double *ob = aligned_alloc(64, (2 * R * count + 64) * sizeof(double));
        double *zin = (double *)((char *)ib + offs[oi]), *zout = (double *)((char *)ob + offs[oi]);
        for (size_t i = 0; i < 2 * R * count; i++) zin[i] = (double)i * 0.001;
        double a = timeit(fs[0], zin, zout, count, offs[oi]);
        double b = timeit(fs[1], zin, zout, count, offs[oi]);
        double c = timeit(fs[2], zin, zout, count, offs[oi]);
        printf("%6zu %4d %9.2f %9.2f %9.2f  %.2f\n", count, offs[oi], a, b, c, c / b);
        free(ib); free(ob);
    }
    return 0;
}
