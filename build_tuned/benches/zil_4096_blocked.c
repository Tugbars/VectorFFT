/* zil_4096_blocked.c — checklist item 15: the BLOCKED z EXECUTOR at N=4096.
 * The split-point recursion (classic engine's shipped blocking; NOT the
 * buffered-tiling slab NO-GO): one global DIF-outer pass, then R_o
 * INDEPENDENT, fully L1-resident inner transforms that complete all their
 * passes while hot.
 *
 *   4096 = 16 x 256, DIF-outer, x[n], n = c*256 + a:
 *   PASS A (global): radix16_z_t2d — legs c at stride 256, columns a
 *     contiguous, radix-16 butterfly then POST-twiddle W_4096^(m*a);
 *     output leg m -> y[m*256 + a]  (contiguous blocks per m).
 *   INNER (x16): block m = 256-pt z transform of y[m*256..], our 8x32
 *     two-pass: leaf radix32b (contiguous) -> scratch; then radix8_z_t2ss:
 *     strided loads (Gs=8) AND strided stores (OLs=512, OGs=16, base m)
 *     so inner bin q lands at X[q*16 + m] — natural order, no extra pass.
 *   Inner twiddles: ONE shared W_256 VTW2 stream (~3.5 KB, L1) for all 16.
 *
 * Gate vs naive-4096 BEFORE timing. Race: blocked vs the flat 64x64
 * two-pass (the 0.38x baseline) vs live MKL-IL, order-rotated best-of-7.
 *
 * Build: python build.py --src benches/zil_4096_blocked.c --mkl
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include <mkl_dfti.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix16_z_t2d_fwd_avx2)   /* outer: post-twiddle DIF pass */
D(radix32_z_n1b_fwd_avx2)   /* inner pass1: blocked radix-32 leaf */
D(radix8_z_t2ss_fwd_avx2)   /* inner pass2: strided loads AND stores */
D(radix64_z_n1b2_fwd_avx2)  /* flat baseline pass1 */
D(radix64_z_t2s_fwd_avx2)   /* flat baseline pass2 */

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free(j);
}

enum { N = 4096, RO = 16, NI = 256, IR1 = 8, IR2 = 32 };

/* VTW2 fill: per column-pair, legs 1..R-1, W_Nw^(l*k) */
static void vtw2_fill(double *tw, int R, int K, int Nw)
{
    for (int p = 0; p < K / 2; p++)
        for (int l = 1; l < R; l++) {
            double *rec = tw + ((size_t)p * (R - 1) + (l - 1)) * 8;
            for (int j = 0; j < 2; j++) {
                int k = 2 * p + j;
                double a = -2.0 * M_PI * (double)((long)l * k) / (double)Nw;
                rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
            }
        }
}

/* blocked executor: z -> z (in-place overall) via yA (outer out) + zscr */
static void run_blocked(double *z, double *yA, double *zscr,
                        const double *twA, const double *twI)
{
    /* PASS A: outer radix-16 DIF + post-twiddle; legs stride NI, cols contig */
    radix16_z_t2d_fwd_avx2(z, 0, yA, 0, twA, 0,
                           NI, 0, NI, 0, NI);
    /* 16 inner 256-pt transforms, L1-resident, strided final stores */
    for (int m = 0; m < RO; m++) {
        const double *blk = yA + (size_t)2 * m * NI;
        radix32_z_n1b_fwd_avx2(blk, 0, zscr, 0, 0, 0,
                               IR1, 0, IR1, 0, IR1);
        radix8_z_t2ss_fwd_avx2(zscr, 0, z + 2 * m, 0, twI, 0,
                               1, IR1, (unsigned long long)(IR2 * RO), RO,
                               IR2);
    }
}

static void run_flat(const double *cs_tw, double *z, double *zscr)
{
    radix64_z_n1b2_fwd_avx2(z, 0, zscr, 0, 0, 0, 64, 0, 64, 0, 64);
    radix64_z_t2s_fwd_avx2(zscr, 0, z, 0, cs_tw, 0, 1, 64, 64, 1, 64);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *yA = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *zscr = (double *)_mm_malloc((size_t)2 * NI * 8, 64);
    double *zscr64 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    /* outer: R=16 over K=NI columns of W_4096 ; inner: R=8 over K=32 of W_256
     * ; flat baseline: R=64 over K=64 of W_4096 */
    double *twA = (double *)_mm_malloc((size_t)(NI / 2) * (RO - 1) * 8 * 8, 64);
    double *twI = (double *)_mm_malloc((size_t)(IR2 / 2) * (IR1 - 1) * 8 * 8, 64);
    double *twF = (double *)_mm_malloc((size_t)(64 / 2) * (64 - 1) * 8 * 8, 64);
    vtw2_fill(twA, RO, NI, N);
    vtw2_fill(twI, IR1, IR2, NI);
    vtw2_fill(twF, 64, 64, N);

    srand(4096);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* gate vs naive */
    memcpy(z, z0, (size_t)2 * N * 8);
    run_blocked(z, yA, zscr, twA, twI);
    double err = 0, mag = 0;
    for (int m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)((long)n * m % N) / (double)N;
            double c = cos(a), s = sin(a);
            sr += z0[2 * n] * c - z0[2 * n + 1] * s;
            si += z0[2 * n] * s + z0[2 * n + 1] * c;
        }
        double d = fabs(z[2 * m] - sr) + fabs(z[2 * m + 1] - si);
        if (d > err) err = d;
        double g = fabs(sr) + fabs(si);
        if (g > mag) mag = g;
    }
    printf("GATE blocked-4096 relerr=%.3e %s\n", err / mag,
           (err / mag < 1e-12) ? "PASS" : "FAIL");
    if (err / mag >= 1e-12) return 1;

    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);

    int reps = 600;
    double best[3] = { 1e18, 1e18, 1e18 };
    memcpy(z, z0, (size_t)2 * N * 8);
    for (int t = 0; t < 7; t++) {
        if (t) cachebust();
        for (int q = 0; q < 3; q++) {
            int a = (t % 3 + q) % 3;   /* rotate start */
            double t0, ns;
            if (a == 0) {
                for (int w = 0; w < 5; w++) run_blocked(z, yA, zscr, twA, twI);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_blocked(z, yA, zscr, twA, twI);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else if (a == 1) {
                for (int w = 0; w < 5; w++) run_flat(twF, z, zscr64);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_flat(twF, z, zscr64);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else {
                for (int w = 0; w < 5; w++) DftiComputeForward(h, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                ns = (now_ms() - t0) * 1e6 / reps;
            }
            if (ns < best[a]) best[a] = ns;
        }
    }
    printf("\n# N=4096: blocked z executor (16 x inner-256) vs flat 64x64 vs MKL-IL\n");
    printf("BLOCKED (t2d + 16x[32b+t2ss]) %10.1f ns   vsMKL %.2f\n", best[0], best[2] / best[0]);
    printf("FLAT    (64x64 two-pass)      %10.1f ns   vsMKL %.2f\n", best[1], best[2] / best[1]);
    printf("MKL-IL                        %10.1f ns\n", best[2]);
    printf("DONE\n");
    return 0;
}
