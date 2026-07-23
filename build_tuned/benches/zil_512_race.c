/* zil_512_race.c — checklist item 10: the FIRST end-to-end IL-native
 * transform. Two-pass z->z N=512 composed from the z family per the §6
 * construction table, three factorizations raced vs MKL-IL live:
 *
 *   pass 1 (twiddle-free): radixR2_z_n1* — R1 contiguous columns, grid
 *     z[n], n = p*R1 + c  (Ls = R1, count = R1)
 *   pass 2 (streamed VTW2): radixR1_z_t2s — R2 STRIDED columns (rows of the
 *     grid; Gs = R1, Ls = 1), twiddles W_512^(l*k), stores natural
 *     X[m*R2 + p] (OLs = R2, OGs = 1).
 *
 * Candidates: 32x16 (R1=32: t2s r32, leaf r16 blocked)  — MKL's own split
 *             16x32 (R1=16: t2s r16, leaf r32 blocked)
 *             8x64  (R1=8:  t2s r8,  leaf r64 8x8 blocked2)
 * Gate: each vs naive-512 (tolerance). Race: ns/transform vs MKL-IL
 * in-place (same process, order-rotated best-of-7, cachebust).
 *
 * Build: python build.py --src benches/zil_512_race.c --mkl
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
D(radix16_z_n1b_fwd_avx2) D(radix32_z_n1b_fwd_avx2) D(radix64_z_n1b2_fwd_avx2)
D(radix8_z_t2s_fwd_avx2)  D(radix16_z_t2s_fwd_avx2) D(radix32_z_t2s_fwd_avx2)

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

enum { N = 512 };

typedef struct {
    const char *nm;
    int R1, R2;          /* N = R1*R2; pass1 radix R2, pass2 radix R1 */
    zfn leaf, t2s;
    double *tw;          /* VTW2 for pass2: per column-pair, legs 1..R1-1 */
} cand_t;

/* two-pass execute: z (in-place overall) via zscr */
static void run_cand(const cand_t *c, double *z, double *zscr)
{
    /* pass1: R1 columns of length R2; z -> zscr, grid preserved */
    c->leaf(z, 0, zscr, 0, 0, 0,
            (unsigned long long)c->R1, 0, (unsigned long long)c->R1, 0,
            (unsigned long long)c->R1);
    /* pass2: R2 strided columns (rows), twiddled, natural out; zscr -> z */
    c->t2s(zscr, 0, z, 0, c->tw, 0,
           /*Ls=*/1, /*Gs=*/(unsigned long long)c->R1,
           /*OLs=*/(unsigned long long)c->R2, /*OGs=*/1,
           (unsigned long long)c->R2);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    cand_t cs[3] = {
        { "32x16 (t2s32,leaf16b)", 32, 16, radix16_z_n1b_fwd_avx2, radix32_z_t2s_fwd_avx2, 0 },
        { "16x32 (t2s16,leaf32b)", 16, 32, radix32_z_n1b_fwd_avx2, radix16_z_t2s_fwd_avx2, 0 },
        { "8x64  (t2s8, leaf64b2)", 8, 64, radix64_z_n1b2_fwd_avx2, radix8_z_t2s_fwd_avx2, 0 },
    };
    /* VTW2 fills: pass2 count-axis k = 0..R2-1, legs l = 1..R1-1, W_512^(l*k) */
    for (int ci = 0; ci < 3; ci++) {
        int R1 = cs[ci].R1, R2 = cs[ci].R2;
        cs[ci].tw = (double *)_mm_malloc((size_t)(R2 / 2) * (R1 - 1) * 8 * 8, 64);
        for (int p = 0; p < R2 / 2; p++)
            for (int l = 1; l < R1; l++) {
                double *rec = cs[ci].tw + ((size_t)p * (R1 - 1) + (l - 1)) * 8;
                for (int j = 0; j < 2; j++) {
                    int k = 2 * p + j;
                    double a = -2.0 * M_PI * (double)(l * k) / (double)N;
                    rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                    rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
                }
            }
    }

    double *z0 = (double *)_mm_malloc(2 * N * 8, 64);
    double *z = (double *)_mm_malloc(2 * N * 8, 64);
    double *zscr = (double *)_mm_malloc(2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc(N * 8, 64), *Ri = (double *)_mm_malloc(N * 8, 64);
    srand(512);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    /* naive reference */
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
    /* gates */
    for (int ci = 0; ci < 3; ci++) {
        memcpy(z, z0, 2 * N * 8);
        run_cand(&cs[ci], z, zscr);
        double err = 0, mag = 0;
        for (int m = 0; m < N; m++) {
            double d = fabs(z[2 * m] - Rr[m]) + fabs(z[2 * m + 1] - Ri[m]);
            if (d > err) err = d;
            double g = fabs(Rr[m]) + fabs(Ri[m]);
            if (g > mag) mag = g;
        }
        printf("GATE %-24s relerr=%.3e %s\n", cs[ci].nm, err / mag,
               (err / mag < 1e-12) ? "PASS" : "FAIL");
        if (err / mag >= 1e-12) return 1;
    }

    /* MKL arm */
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);

    int reps = 4000;
    double best[4] = { 1e18, 1e18, 1e18, 1e18 };
    memcpy(z, z0, 2 * N * 8);
    for (int t = 0; t < 7; t++) {
        if (t) cachebust();
        for (int q = 0; q < 4; q++) {
            int a = (t & 1) ? 3 - q : q;
            if (a < 3) {
                for (int w = 0; w < 10; w++) run_cand(&cs[a], z, zscr);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) run_cand(&cs[a], z, zscr);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < best[a]) best[a] = ns;
            } else {
                for (int w = 0; w < 10; w++) DftiComputeForward(h, z);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < best[3]) best[3] = ns;
            }
        }
    }
    printf("\n# N=512 z->z two-pass vs MKL-IL (in-place, same process)\n");
    for (int ci = 0; ci < 3; ci++)
        printf("Z512 %-24s %8.1f ns   vsMKL %.2f\n", cs[ci].nm, best[ci],
               best[3] / best[ci]);
    printf("MKL-IL                        %8.1f ns\n", best[3]);
    printf("DONE\n");
    return 0;
}
