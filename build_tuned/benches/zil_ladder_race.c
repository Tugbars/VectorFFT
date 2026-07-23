/* zil_ladder_race.c — the INTERIM pure-IL scoreboard: z-native two-pass at
 * every composable N (64..4096) vs LIVE MKL-IL, same process, in-place.
 * One (uncalibrated, construction-law-favored) factorization per cell:
 * biggest leaf + smallest t2s radix available. Each cell gated vs naive-N
 * BEFORE timing. Methodology: pinned P-core, best-of-7, cachebust, order
 * flipped (ours vs MKL) per trial.
 *
 * Build: python build.py --src benches/zil_ladder_race.c --mkl
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

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long);
#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix8_z_n1_fwd_avx2)   D(radix16_z_n1b_fwd_avx2) D(radix32_z_n1b_fwd_avx2)
D(radix64_z_n1b2_fwd_avx2)
D(radix8_z_t2s_fwd_avx2)  D(radix16_z_t2s_fwd_avx2) D(radix32_z_t2s_fwd_avx2)
D(radix64_z_t2s_fwd_avx2)

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

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    struct { int N, R1, R2; zfn leaf, t2s; const char *nm; } cells[] = {
        { 64,   8,  8,  radix8_z_n1_fwd_avx2,    radix8_z_t2s_fwd_avx2,  "8x8"   },
        { 128,  8,  16, radix16_z_n1b_fwd_avx2,  radix8_z_t2s_fwd_avx2,  "8x16"  },
        { 256,  8,  32, radix32_z_n1b_fwd_avx2,  radix8_z_t2s_fwd_avx2,  "8x32"  },
        { 512,  8,  64, radix64_z_n1b2_fwd_avx2, radix8_z_t2s_fwd_avx2,  "8x64"  },
        { 1024, 16, 64, radix64_z_n1b2_fwd_avx2, radix16_z_t2s_fwd_avx2, "16x64" },
        { 2048, 32, 64, radix64_z_n1b2_fwd_avx2, radix32_z_t2s_fwd_avx2, "32x64" },
        { 4096, 64, 64, radix64_z_n1b2_fwd_avx2, radix64_z_t2s_fwd_avx2, "64x64" },
    };
    int nc = 7;

    printf("# INTERIM pure IL vs IL: z-native two-pass (uncalibrated, one shape/cell) vs live MKL-IL\n");
    printf("%-6s %-7s %10s %10s %8s   gate\n", "N", "shape", "Z(ns)", "MKL(ns)", "Z/MKL");
    for (int ci = 0; ci < nc; ci++) {
        int N = cells[ci].N, R1 = cells[ci].R1, R2 = cells[ci].R2;
        double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        double *z = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        double *zs = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        double *tw = (double *)_mm_malloc((size_t)(R2 / 2) * (R1 - 1) * 8 * 8, 64);
        srand(9 + N);
        for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
        for (int p = 0; p < R2 / 2; p++)
            for (int l = 1; l < R1; l++) {
                double *rec = tw + ((size_t)p * (R1 - 1) + (l - 1)) * 8;
                for (int j = 0; j < 2; j++) {
                    int k = 2 * p + j;
                    double a = -2.0 * M_PI * (double)((long)l * k) / (double)N;
                    rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                    rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
                }
            }
        /* gate vs naive */
        memcpy(z, z0, (size_t)2 * N * 8);
        cells[ci].leaf(z, 0, zs, 0, 0, 0, (unsigned long long)R1, 0,
                       (unsigned long long)R1, 0, (unsigned long long)R1);
        cells[ci].t2s(zs, 0, z, 0, tw, 0, 1, (unsigned long long)R1,
                      (unsigned long long)R2, 1, (unsigned long long)R2);
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
        int gate_ok = (err / mag < 1e-12);

        DFTI_DESCRIPTOR_HANDLE h = NULL;
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiCommitDescriptor(h);
        int reps = (int)(2e6 / (double)N); if (reps < 200) reps = 200;
        double bz = 1e18, bm = 1e18;
        memcpy(z, z0, (size_t)2 * N * 8);
        for (int t = 0; t < 7; t++) {
            if (t) cachebust();
            for (int q = 0; q < 2; q++) {
                int a = (t & 1) ? 1 - q : q;
                if (a == 0) {
                    for (int w = 0; w < 10; w++) {
                        cells[ci].leaf(z, 0, zs, 0, 0, 0, (unsigned long long)R1, 0,
                                       (unsigned long long)R1, 0, (unsigned long long)R1);
                        cells[ci].t2s(zs, 0, z, 0, tw, 0, 1, (unsigned long long)R1,
                                      (unsigned long long)R2, 1, (unsigned long long)R2);
                    }
                    double t0 = now_ms();
                    for (int i = 0; i < reps; i++) {
                        cells[ci].leaf(z, 0, zs, 0, 0, 0, (unsigned long long)R1, 0,
                                       (unsigned long long)R1, 0, (unsigned long long)R1);
                        cells[ci].t2s(zs, 0, z, 0, tw, 0, 1, (unsigned long long)R1,
                                      (unsigned long long)R2, 1, (unsigned long long)R2);
                    }
                    double ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bz) bz = ns;
                } else {
                    for (int w = 0; w < 10; w++) DftiComputeForward(h, z);
                    double t0 = now_ms();
                    for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                    double ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bm) bm = ns;
                }
            }
        }
        printf("%-6d %-7s %10.1f %10.1f %8.2f   %s\n", N, cells[ci].nm, bz, bm,
               bm / bz, gate_ok ? "PASS" : "FAIL");
        DftiFreeDescriptor(&h);
        _mm_free(z0); _mm_free(z); _mm_free(zs); _mm_free(tw);
    }
    printf("DONE\n");
    return 0;
}
