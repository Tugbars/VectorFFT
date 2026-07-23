/* zil_blocked_race.c — checklist item 14 DECIDED BY MEASUREMENT (user
 * directive): monolithic vs BLOCKED (Tier-B analog: PASS-1 halves parked to
 * function-scope zspill, PASS-2 reload-on-demand combine) for the z-native
 * family, per radix, K sweep.
 *
 * Arms: n1 r16/r32/r64 mono-vs-blocked; t2 r16/r32 mono-vs-blocked (VTW2
 * stream filled once per (R,K)). Methodology: pinned P-core logical 2, HIGH
 * prio, best-of-7, 32MB cachebust between trials, arm order flipped per
 * trial. ns per R-point transform.
 *
 * Build: python build.py --src benches/zil_blocked_race.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>   /* _mm_malloc/_mm_free prototypes (x64: no implicit int!) */
#include <windows.h>

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
D(radix16_z_n1_fwd_avx2)  D(radix16_z_n1b_fwd_avx2)
D(radix32_z_n1_fwd_avx2)  D(radix32_z_n1b_fwd_avx2)
D(radix64_z_n1_fwd_avx2)  D(radix64_z_n1b_fwd_avx2)
D(radix64_z_n1b2_fwd_avx2)
D(radix16_z_t2_fwd_avx2)  D(radix16_z_t2b_fwd_avx2)
D(radix32_z_t2_fwd_avx2)  D(radix32_z_t2b_fwd_avx2)

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

    struct { const char *nm; int R; int tw; zfn mono, blk; } cells[] = {
        { "n1 r16", 16, 0, radix16_z_n1_fwd_avx2, radix16_z_n1b_fwd_avx2 },
        { "n1 r32", 32, 0, radix32_z_n1_fwd_avx2, radix32_z_n1b_fwd_avx2 },
        { "n1 r64", 64, 0, radix64_z_n1_fwd_avx2, radix64_z_n1b_fwd_avx2 },
        { "r64 8x8", 64, 0, radix64_z_n1_fwd_avx2, radix64_z_n1b2_fwd_avx2 },
        { "t2 r16", 16, 1, radix16_z_t2_fwd_avx2, radix16_z_t2b_fwd_avx2 },
        { "t2 r32", 32, 1, radix32_z_t2_fwd_avx2, radix32_z_t2b_fwd_avx2 },
    };
    int Ks[] = { 16, 64, 256, 1024, 4096 };

    printf("# monolithic vs blocked z kernels — ns/transform, best-of-7 rotated\n");
    printf("%-8s %-6s %10s %10s %9s\n", "kernel", "K", "MONO", "BLOCKED", "delta");
    for (int ci = 0; ci < 6; ci++) {
        int R = cells[ci].R;
        for (int ki = 0; ki < 5; ki++) {
            int K = Ks[ki];
            size_t S = (size_t)2 * K;
            double *zin = (double *)_mm_malloc(R * S * 8, 64);
            double *zo1 = (double *)_mm_malloc(R * S * 8, 64);
            double *zo2 = (double *)_mm_malloc(R * S * 8, 64);
            double *tw = 0;
            srand(7 + R + K);
            for (size_t i = 0; i < (size_t)R * S; i++)
                zin[i] = (double)rand() / RAND_MAX - 0.5;
            if (cells[ci].tw) {
                int N = R * K;
                tw = (double *)_mm_malloc((size_t)(K / 2) * (R - 1) * 8 * 8, 64);
                for (int p = 0; p < K / 2; p++)
                    for (int l = 1; l < R; l++) {
                        double *rec = tw + ((size_t)p * (R - 1) + (l - 1)) * 8;
                        for (int j = 0; j < 2; j++) {
                            int k = 2 * p + j;
                            double a = -2.0 * M_PI * (double)(l * k) / (double)N;
                            rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                            rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
                        }
                    }
            }
            /* correctness cross-check: mono vs blocked outputs must agree
             * to tolerance (different op order -> not bit) */
            cells[ci].mono(zin, 0, zo1, 0, tw, 0, K, 0, K, 0, K);
            cells[ci].blk(zin, 0, zo2, 0, tw, 0, K, 0, K, 0, K);
            double xerr = 0;
            for (size_t i = 0; i < (size_t)R * S; i++) {
                double d = fabs(zo1[i] - zo2[i]);
                if (d > xerr) xerr = d;
            }
            if (xerr > 1e-11 * R) {
                printf("%-8s %-6d CROSS-CHECK FAIL %.2e\n", cells[ci].nm, K, xerr);
                return 1;
            }
            int reps = (int)(4e6 / ((double)K * R)); if (reps < 100) reps = 100;
            double best[2] = { 1e18, 1e18 };
            for (int t = 0; t < 7; t++) {
                if (t) cachebust();
                for (int a = 0; a < 2; a++) {
                    int arm = (t & 1) ? 1 - a : a;
                    zfn f = arm ? cells[ci].blk : cells[ci].mono;
                    for (int w = 0; w < 10; w++) f(zin, 0, zo1, 0, tw, 0, K, 0, K, 0, K);
                    double t0 = now_ms();
                    for (int i = 0; i < reps; i++) f(zin, 0, zo1, 0, tw, 0, K, 0, K, 0, K);
                    double ns = (now_ms() - t0) * 1e6 / ((double)reps * K);
                    if (ns < best[arm]) best[arm] = ns;
                }
            }
            printf("%-8s %-6d %10.3f %10.3f %+8.1f%%\n", cells[ci].nm, K,
                   best[0], best[1], 100.0 * (best[1] - best[0]) / best[0]);
            _mm_free(zin); _mm_free(zo1); _mm_free(zo2); if (tw) _mm_free(tw);
        }
    }
    printf("DONE\n");
    return 0;
}
