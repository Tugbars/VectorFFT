/* zil_4096_decomp.c — cost decomposition of OUR flat 4096 (64x64): time each
 * pass alone, plus a twiddle-traffic isolation arm (t2s with a TINY looped
 * table substituted — WRONG results, measures the VTW2-stream cost only).
 * Feeds docs/research/high_n_loss_analysis.md.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix64_z_n1b2_fwd_avx2) D(radix64_z_t2s_fwd_avx2)

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

enum { N = 4096 };

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    double *z = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *zs = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *tw = (double *)_mm_malloc((size_t)32 * 63 * 8 * 8, 64);   /* full 126KB */
    double *twS = (double *)_mm_malloc((size_t)1 * 63 * 8 * 8, 64);   /* 1 col-pair ~4KB */
    srand(1);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;
    for (int p = 0; p < 32; p++)
        for (int l = 1; l < 64; l++) {
            double *rec = tw + ((size_t)p * 63 + (l - 1)) * 8;
            for (int j = 0; j < 2; j++) {
                int k = 2 * p + j;
                double a = -2.0 * M_PI * (double)((long)l * k) / (double)N;
                rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
            }
        }
    memcpy(twS, tw, (size_t)63 * 8 * 8);  /* one col-pair, reused (WRONG math, traffic probe) */

    int reps = 1500;
    double b[4] = { 1e18, 1e18, 1e18, 1e18 };
    for (int t = 0; t < 7; t++) {
        if (t) cachebust();
        for (int q = 0; q < 4; q++) {
            int a = (t % 4 + q) % 4;
            double t0, ns;
            if (a == 0) {          /* pass 1 alone: 8x8-blocked leaf */
                for (int w = 0; w < 5; w++) radix64_z_n1b2_fwd_avx2(z, 0, zs, 0, 0, 0, 64, 0, 64, 0, 64);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) radix64_z_n1b2_fwd_avx2(z, 0, zs, 0, 0, 0, 64, 0, 64, 0, 64);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else if (a == 1) {   /* pass 2 alone: t2s r64, full VTW2 stream */
                for (int w = 0; w < 5; w++) radix64_z_t2s_fwd_avx2(zs, 0, z, 0, tw, 0, 1, 64, 64, 1, 64);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) radix64_z_t2s_fwd_avx2(zs, 0, z, 0, tw, 0, 1, 64, 64, 1, 64);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else if (a == 2) {
                /* pass 2 with a TINY reused table (WRONG math, isolates the
                 * VTW2-stream cost): count=2 sub-calls restart the internal
                 * cursor onto the same 4KB record set; zin/zout offsets keep
                 * the REAL data traffic (actual columns c,c+1 each call). */
                for (int w = 0; w < 5; w++)
                    for (int c = 0; c < 64; c += 2)
                        radix64_z_t2s_fwd_avx2(zs + 2 * 64 * c, 0, z + 2 * c, 0,
                                               twS, 0, 1, 64, 64, 1, 2);
                t0 = now_ms();
                for (int i = 0; i < reps; i++)
                    for (int c = 0; c < 64; c += 2)
                        radix64_z_t2s_fwd_avx2(zs + 2 * 64 * c, 0, z + 2 * c, 0,
                                               twS, 0, 1, 64, 64, 1, 2);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else {               /* both passes (the flat champion) */
                for (int w = 0; w < 5; w++) {
                    radix64_z_n1b2_fwd_avx2(z, 0, zs, 0, 0, 0, 64, 0, 64, 0, 64);
                    radix64_z_t2s_fwd_avx2(zs, 0, z, 0, tw, 0, 1, 64, 64, 1, 64);
                }
                t0 = now_ms();
                for (int i = 0; i < reps; i++) {
                    radix64_z_n1b2_fwd_avx2(z, 0, zs, 0, 0, 0, 64, 0, 64, 0, 64);
                    radix64_z_t2s_fwd_avx2(zs, 0, z, 0, tw, 0, 1, 64, 64, 1, 64);
                }
                ns = (now_ms() - t0) * 1e6 / reps;
            }
            if (ns < b[a]) b[a] = ns;
        }
    }
    printf("# OUR flat 4096 (64x64) decomposition, ns/transform\n");
    printf("pass1 leaf64-8x8 alone      %9.1f\n", b[0]);
    printf("pass2 t2s64 full-126KB-tw   %9.1f\n", b[1]);
    printf("pass2 t2s64 tiny-tw (probe) %9.1f   (tw-stream cost ~ %.1f)\n", b[2], b[1] - b[2]);
    printf("both passes (champion)      %9.1f   (sum of parts %.1f)\n", b[3], b[0] + b[1]);
    printf("DONE\n");
    return 0;
}
