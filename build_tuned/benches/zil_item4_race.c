/* zil_item4_race.c — checklist item 4 MEASURED: corner-turn-in-stores.
 *
 * Two ways to bridge pass1 -> pass2 in the z two-pass:
 *   CHAMP (current): plain leaf stores (grid preserved) + t2s pass2 with
 *          STRIDED 2x128 loads (the tax).
 *   T-shape (item 4): n1t leaf — transposed stores via vperm2f128 pairs
 *          (full-width, sectioned) -> pass2 is PLAIN t2 (contiguous loads,
 *          zero strided anything).
 * Per N: current-champion shape vs 2 transposed shapes, gate vs naive,
 * live MKL-IL arm, order-rotated best-of-7.
 *
 * Transposed composition (N = R1*R2, leaf radix R2 MONOLITHIC n1t):
 *   pass1: n1t  (Ls=R1 count=R1, OLs=R2 -> scratch[c*R2 + p])
 *   pass2: t2   (Ls=R2 count=R2, OLs=R2 OGs=1 -> X[m*R2 + p]) VTW2 W_N^(l*k)
 *
 * Build: python build.py --src benches/zil_item4_race.c --mkl
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
/* champion pieces */
D(radix32_z_n1b_fwd_avx2) D(radix64_z_n1b2_fwd_avx2)
D(radix8_z_t2s_fwd_avx2)  D(radix16_z_t2s_fwd_avx2) D(radix32_z_t2s_fwd_avx2)
/* item-4 pieces */
D(radix8_z_n1t_fwd_avx2)  D(radix16_z_n1t_fwd_avx2)
D(radix32_z_n1t_fwd_avx2) D(radix64_z_n1t_fwd_avx2)
D(radix8_z_t2_fwd_avx2)   D(radix16_z_t2_fwd_avx2)
D(radix32_z_t2_fwd_avx2)  D(radix64_z_t2_fwd_avx2)

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

typedef struct {
    const char *nm;
    int R1, R2;     /* N = R1*R2; leaf radix R2, pass2 radix R1 */
    zfn leaf, p2;
    int trans;      /* 1: n1t + plain t2 ; 0: champ leaf + t2s */
    double *tw;
} arm_t;

static void run_arm(const arm_t *a, int N, double *z, double *zs)
{
    (void)N;
    if (a->trans) {
        a->leaf(z, 0, zs, 0, 0, 0, (unsigned long long)a->R1, 0,
                (unsigned long long)a->R2, 0, (unsigned long long)a->R1);
        a->p2(zs, 0, z, 0, a->tw, 0, (unsigned long long)a->R2, 0,
              (unsigned long long)a->R2, 1, (unsigned long long)a->R2);
    } else {
        a->leaf(z, 0, zs, 0, 0, 0, (unsigned long long)a->R1, 0,
                (unsigned long long)a->R1, 0, (unsigned long long)a->R1);
        a->p2(zs, 0, z, 0, a->tw, 0, 1, (unsigned long long)a->R1,
              (unsigned long long)a->R2, 1, (unsigned long long)a->R2);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    struct { int N; arm_t arms[3]; int na; } cells[] = {
      { 256, {
        { "C 8x32 (32b+t2s8)", 8, 32, radix32_z_n1b_fwd_avx2, radix8_z_t2s_fwd_avx2, 0, 0 },
        { "T 32x8 (n1t8+t2-32)", 32, 8, radix8_z_n1t_fwd_avx2, radix32_z_t2_fwd_avx2, 1, 0 },
        { "T 16x16 (n1t16+t2-16)", 16, 16, radix16_z_n1t_fwd_avx2, radix16_z_t2_fwd_avx2, 1, 0 },
      }, 3 },
      { 512, {
        { "C 8x64 (64b2+t2s8)", 8, 64, radix64_z_n1b2_fwd_avx2, radix8_z_t2s_fwd_avx2, 0, 0 },
        { "T 64x8 (n1t8+t2-64)", 64, 8, radix8_z_n1t_fwd_avx2, radix64_z_t2_fwd_avx2, 1, 0 },
        { "T 16x32 (n1t32+t2-16)", 16, 32, radix32_z_n1t_fwd_avx2, radix16_z_t2_fwd_avx2, 1, 0 },
      }, 3 },
      { 1024, {
        { "C 16x64 (64b2+t2s16)", 16, 64, radix64_z_n1b2_fwd_avx2, radix16_z_t2s_fwd_avx2, 0, 0 },
        { "T 32x32 (n1t32+t2-32)", 32, 32, radix32_z_n1t_fwd_avx2, radix32_z_t2_fwd_avx2, 1, 0 },
        { "T 16x64 (n1t64+t2-16)", 16, 64, radix64_z_n1t_fwd_avx2, radix16_z_t2_fwd_avx2, 1, 0 },
      }, 3 },
      { 2048, {
        { "C 32x64 (64b2+t2s32)", 32, 64, radix64_z_n1b2_fwd_avx2, radix32_z_t2s_fwd_avx2, 0, 0 },
        { "T 32x64 (n1t64+t2-32)", 32, 64, radix64_z_n1t_fwd_avx2, radix32_z_t2_fwd_avx2, 1, 0 },
        { "T 64x32 (n1t32+t2-64)", 64, 32, radix32_z_n1t_fwd_avx2, radix64_z_t2_fwd_avx2, 1, 0 },
      }, 3 },
    };

    for (int ci = 0; ci < 4; ci++) {
        int N = cells[ci].N, na = cells[ci].na;
        double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        double *z = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        double *zs = (double *)_mm_malloc((size_t)2 * N * 8, 64);
        srand(44 + N);
        for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
        /* naive ref */
        double *Rr = (double *)malloc(N * 8), *Ri = (double *)malloc(N * 8);
        for (int m = 0; m < N; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double a = -2.0 * M_PI * (double)((long)n * m % N) / (double)N;
                double c = cos(a), s = sin(a);
                sr += z0[2 * n] * c - z0[2 * n + 1] * s;
                si += z0[2 * n] * s + z0[2 * n + 1] * c;
            }
            Rr[m] = sr; Ri[m] = si;
        }
        int ok = 1;
        for (int ai = 0; ai < na; ai++) {
            arm_t *a = &cells[ci].arms[ai];
            a->tw = (double *)_mm_malloc((size_t)(a->R2 / 2) * (a->R1 - 1) * 8 * 8, 64);
            vtw2_fill(a->tw, a->R1, a->R2, N);
            memcpy(z, z0, (size_t)2 * N * 8);
            run_arm(a, N, z, zs);
            double err = 0, mag = 0;
            for (int m = 0; m < N; m++) {
                double d = fabs(z[2 * m] - Rr[m]) + fabs(z[2 * m + 1] - Ri[m]);
                if (d > err) err = d;
                double g = fabs(Rr[m]) + fabs(Ri[m]);
                if (g > mag) mag = g;
            }
            if (err / mag >= 1e-12) {
                printf("GATE N=%d %-22s FAIL %.2e\n", N, a->nm, err / mag);
                ok = 0;
            }
        }
        if (!ok) return 1;

        DFTI_DESCRIPTOR_HANDLE h = NULL;
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiCommitDescriptor(h);
        int reps = (int)(2e6 / (double)N); if (reps < 300) reps = 300;
        double best[4] = { 1e18, 1e18, 1e18, 1e18 };
        int nA = na + 1;
        memcpy(z, z0, (size_t)2 * N * 8);
        for (int t = 0; t < 7; t++) {
            if (t) cachebust();
            for (int q = 0; q < nA; q++) {
                int a = (t % nA + q) % nA;
                double t0, ns;
                if (a < na) {
                    for (int w = 0; w < 10; w++) run_arm(&cells[ci].arms[a], N, z, zs);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) run_arm(&cells[ci].arms[a], N, z, zs);
                    ns = (now_ms() - t0) * 1e6 / reps;
                } else {
                    for (int w = 0; w < 10; w++) DftiComputeForward(h, z);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                    ns = (now_ms() - t0) * 1e6 / reps;
                }
                if (ns < best[a]) best[a] = ns;
            }
        }
        printf("\n# N=%d (MKL-IL %.1f ns)\n", N, best[na]);
        for (int ai = 0; ai < na; ai++)
            printf("  %-24s %9.1f ns   vsMKL %.2f\n", cells[ci].arms[ai].nm,
                   best[ai], best[na] / best[ai]);
        DftiFreeDescriptor(&h);
        for (int ai = 0; ai < na; ai++) _mm_free(cells[ci].arms[ai].tw);
        _mm_free(z0); _mm_free(z); _mm_free(zs); free(Rr); free(Ri);
    }
    printf("DONE\n");
    return 0;
}
