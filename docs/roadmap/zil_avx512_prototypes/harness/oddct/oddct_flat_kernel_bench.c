#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D(n) void n(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
D(radix25_z_t2cp_fwd_avx512) D(radix25_z_t2cp_ct_fwd_avx512) D(radix25_z_n1c_fwd_avx512) D(radix25_z_n1c_ct_fwd_avx512)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static void bench(const char *nm, kfn a, kfn b, int t2cp, size_t C)
{
    const int R = 25; const size_t Dg = 3, Gs = C, Ls = t2cp ? Dg * C : C, OGs = t2cp ? Dg : 0, n = 2 * R * Ls + 64;
    double *x = malloc(8*n), *ya = malloc(8*n), *yb = malloc(8*n), *tw = malloc(8 * Dg * (R - 1) * 16);
    for (size_t i = 0; i < n; i++) x[i] = sin(0.37 * i) + 0.25 * cos(1.3 * i);
    for (size_t d = 0; d < Dg; d++) for (int l = 1; l < R; l++) { double *r = tw + (d * (R - 1) + l - 1) * 16; double a2 = -2 * M_PI * l * (d + 1) / 75.0, c = cos(a2), s = sin(a2);
        for (int j = 0; j < 4; j++) { r[2*j] = r[2*j+1] = c; r[8+2*j] = -s; r[8+2*j+1] = s; } }
    #define RUN(f, y) do { if (t2cp) { memcpy(y, x, 8*n); f(y, 0, y, 0, tw, 0, Ls, Gs, Ls, OGs, C); } else f(x, 0, y, 0, 0, 0, Ls, 0, Ls, 0, C); } while (0)
    RUN(a, ya); RUN(b, yb);
    double num = 0, den = 0; for (size_t i = 0; i < 2 * R * Ls; i++) { num += (ya[i]-yb[i])*(ya[i]-yb[i]); den += ya[i]*ya[i]; }
    const int reps = 2000; double ta[21], tb[21];
    for (int r = 0; r < 21; r++) for (int k = 0; k < 2; k++) { kfn f = k ? b : a; double *y = k ? yb : ya; double best = 1e30;
        for (int q = 0; q < 5; q++) { double t0 = now(); for (int i = 0; i < reps; i++) RUN(f, y); double t = (now() - t0) / reps; if (t < best) best = t; }
        (k ? tb : ta)[r] = best; }
    qsort(ta, 21, 8, cmpd); qsort(tb, 21, 8, cmpd);
    printf("%-5s radix 25 count %3zu: direct %8.0f ns | factored (oddct) %8.0f ns | direct/factored %.2fx | rel diff %.1e\n", nm, C, ta[10], tb[10], ta[10] / tb[10], sqrt(num / den));
    free(x); free(ya); free(yb); free(tw);
}
int main(void)
{
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2, &cs); sched_setaffinity(0, sizeof cs, &cs);
    size_t Cs[] = { 8, 45, 64, 125 };
    for (int i = 0; i < 4; i++) bench("t2cp", radix25_z_t2cp_fwd_avx512, radix25_z_t2cp_ct_fwd_avx512, 1, Cs[i]);
    for (int i = 0; i < 4; i++) bench("n1c", radix25_z_n1c_fwd_avx512, radix25_z_n1c_ct_fwd_avx512, 0, Cs[i]);
    return 0;
}
