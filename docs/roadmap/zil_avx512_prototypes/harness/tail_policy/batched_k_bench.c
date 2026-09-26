/* batched N=1024 = 32x32, K transforms interleaved (z[n*K+k]); stage A: 32 x n1c r32
 * (legs at 32K, count K), stage B: one t2c r32 over 32 digits (legs at K, digit pitch 32K,
 * broadcast per-digit twiddles, count K). Every call's count is K: the tail runs when K%4. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D(n) void n(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
D(radix32_z_n1c_fwd_avx512_narrowfix) D(radix32_z_n1c_fwd_avx512_ladder) D(radix32_z_n1c_fwd_avx512_masked) D(radix32_z_n1c_fwd_avx512_ladder_m3) D(radix32_z_n1c_fwd_avx2)
D(radix32_z_t2c_fwd_avx512_narrowfix) D(radix32_z_t2c_fwd_avx512_ladder) D(radix32_z_t2c_fwd_avx512_masked) D(radix32_z_t2c_fwd_avx512_ladder_m3) D(radix32_z_t2c_fwd_avx2)
enum { NA = 5 };
static const char *nm[NA] = { "narrowfix", "ladder", "masked", "ladder_m3", "avx2" };
static kfn A[NA] = { radix32_z_n1c_fwd_avx512_narrowfix, radix32_z_n1c_fwd_avx512_ladder, radix32_z_n1c_fwd_avx512_masked, radix32_z_n1c_fwd_avx512_ladder_m3, radix32_z_n1c_fwd_avx2 };
static kfn B[NA] = { radix32_z_t2c_fwd_avx512_narrowfix, radix32_z_t2c_fwd_avx512_ladder, radix32_z_t2c_fwd_avx512_masked, radix32_z_t2c_fwd_avx512_ladder_m3, radix32_z_t2c_fwd_avx2 };
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double *table(int vw)   /* per digit d, per leg l: broadcast [c x vw][-s,+s ...] of w^(d*l), N=1024 */
{
    double *t = aligned_alloc(64, sizeof(double) * 32 * 31 * 2 * vw);
    for (int d = 0; d < 32; d++) for (int l = 1; l < 32; l++) {
        double a = -2 * M_PI * d * l / 1024.0, c = cos(a), s = sin(a), *r = t + ((size_t)d * 31 + l - 1) * 2 * vw;
        for (int j = 0; j < vw / 2; j++) { r[2*j] = r[2*j+1] = c; r[vw+2*j] = -s; r[vw+2*j+1] = s; }
    }
    return t;
}
static void run(int a, double *z, const double *tw, int K)
{
    for (int n2 = 0; n2 < 32; n2++) A[a](z + 2 * n2 * K, 0, z + 2 * n2 * K, 0, 0, 0, 32 * K, 0, 32 * K, 0, K);
    B[a](z, 0, z, 0, tw, 0, K, 32 * K, K, 32, K);
}
int main(void)
{
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2, &cs); sched_setaffinity(0, sizeof cs, &cs);
    static const int Ks[] = { 1, 3, 5, 6, 7, 8, 9, 10, 13, 16, 33, 63 };
    double *tw8 = table(8), *tw4 = table(4);
    printf("%-4s %-4s", "K", "K%4");
    for (int a = 0; a < NA; a++) printf(" %11s", nm[a]);
    printf("   (ns per transform)  bitwise\n");
    for (size_t i = 0; i < sizeof Ks / sizeof *Ks; i++) {
        const int K = Ks[i], n = 2 * 1024 * K;
        double *x = aligned_alloc(64, 8 * n + 64), *z = aligned_alloc(64, 8 * n + 64), *ref = malloc(8 * n);
        for (int j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
        int same = 1;
        for (int a = 0; a < NA; a++) { memcpy(z, x, 8 * n); run(a, z, a == 4 ? tw4 : tw8, K); if (!a) memcpy(ref, z, 8 * n); else if (memcmp(ref, z, 8 * n)) same = 0; }
        const int reps = 20000 / K / 8 + 20, R = 21; double t[NA][32];
        for (int r = 0; r < R; r++) for (int kk = 0; kk < NA; kk++) {
            int a = (r & 1) ? NA - 1 - kk : kk; double best = 1e30;
            for (int b = 0; b < 5; b++) { memcpy(z, x, 8 * n); double t0 = now(); for (int q = 0; q < reps; q++) run(a, z, a == 4 ? tw4 : tw8, K); double tt = (now() - t0) / reps; if (tt < best) best = tt; }
            t[a][r] = best / K;
        }
        printf("%-4d %-4d", K, K % 4);
        for (int a = 0; a < NA; a++) { qsort(t[a], R, sizeof(double), cmpd); printf(" %11.0f", t[a][R / 2]); }
        printf("   %s\n", same ? "yes" : "NO");
        free(x); free(z); free(ref);
    }
    return 0;
}
