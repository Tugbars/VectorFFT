/* oddframe_probe.c -- the odd blocked kernels' frame variants in both stack
 * states (2026-09-28). Per (radix, kind, count): base / roll (rolled term loops +
 * 64-B S[]) / p1 (pair-order pass 1) / both, each called with the caller's stack
 * at shift 0 and 16 (the two states rsp mod 32 can hand a Win64 kernel); L1-hot,
 * core 2 HIGH + sibling guard, 21 rounds alternating, min of 5 batches, median.
 * Outputs compared BITWISE to base (same count, same input). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <windows.h>
#include "sibling_guard.h"
#define KARGS const double *, const double *, double *, double *, const double *, const double *, \
              size_t, size_t, size_t, size_t, size_t
typedef void (*kfn)(KARGS);
#define RADS(X) X(9) X(11) X(13) X(15) X(17) X(19) X(21) X(23) X(25) X(27) X(29) X(31) X(37) X(41) X(43) X(47)
#define KD(R, K, V) void radix##R##_z_##K##_fwd_avx2_##V(KARGS);
#define DR(R) KD(R, n1, base) KD(R, n1, roll) KD(R, n1, p1) KD(R, n1, both) KD(R, t2, base) KD(R, t2, roll) KD(R, t2, p1) KD(R, t2, both)
RADS(DR)
enum { NV = 4 };
static const char *VN[NV] = { "base", "roll", "p1", "both" };
#define ER(R) { R, { { radix##R##_z_n1_fwd_avx2_base, radix##R##_z_n1_fwd_avx2_roll, radix##R##_z_n1_fwd_avx2_p1, radix##R##_z_n1_fwd_avx2_both }, \
                     { radix##R##_z_t2_fwd_avx2_base, radix##R##_z_t2_fwd_avx2_roll, radix##R##_z_t2_fwd_avx2_p1, radix##R##_z_t2_fwd_avx2_both } } },
static const struct { int R; kfn f[2][NV]; } KT[] = { RADS(ER) };

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double *vtw2(int R, int cols, int N)
{
    const size_t np = ((size_t)cols + 1) / 2;
    double *t = _aligned_malloc(sizeof(double) * (np * (R - 1) * 8 + 8), 64);
    for (size_t pp = 0; pp < np; pp++)
        for (int l = 1; l < R; l++)
        {
            double *r = t + (pp * (R - 1) + (l - 1)) * 8;
            for (int j = 0; j < 2; j++)
            {
                const double a = -2.0 * 3.14159265358979323846 * (double)l * (double)(2 * pp + j) / (double)N;
                r[2 * j] = r[2 * j + 1] = cos(a);
                r[4 + 2 * j] = -sin(a);
                r[4 + 2 * j + 1] = sin(a);
            }
        }
    return t;
}
static __attribute__((noinline)) double timed(kfn f, const double *x, double *y, const double *tw, size_t c, int reps)
{
    double best = 1e30;
    for (int b = 0; b < 5; b++)
    {
        const double t0 = now_ns();
        for (int i = 0; i < reps; i++) f(x, 0, y, 0, tw, 0, c, 0, c, 0, c);
        const double t = (now_ns() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}
/* the call with the caller's stack moved by `shift` bytes */
static __attribute__((noinline)) double shifted(int shift, kfn f, const double *x, double *y, const double *tw, size_t c, int reps)
{
    volatile char *pad = alloca(64 + shift);
    pad[0] = 0;
    return timed(f, x, y, tw, c, reps);
}

int main(int argc, char **argv)
{
    FILE *lg = fopen(argc > 1 ? argv[1] : "oddframe_results.log", "w");
    bench_pin_caller(2);
    bench_guard_sibling(2);
    static const int CNT[] = { 24, 25, 5 };
    fprintf(lg, "odd blocked frame variants: ns per call at stack state 0 / 16; spread = max/min of the two\n");
    double sum_worst[NV][3] = { { 0 } }, sum_best[NV][3] = { { 0 } };
    int ncell[3] = { 0 };
    for (int ci = 0; ci < 3; ci++)
        for (size_t ri = 0; ri < sizeof KT / sizeof *KT; ri++)
            for (int kind = 0; kind < 2; kind++)
            {
                const int R = KT[ri].R, c = CNT[ci];
                const size_t n = (size_t)R * c;
                double *x = _aligned_malloc(16 * n + 128, 64), *y = _aligned_malloc(16 * n + 128, 64), *ref = malloc(16 * n);
                const double *tw = kind ? vtw2(R, c, (int)n) : 0;
                for (size_t j = 0; j < 2 * n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
                int same[NV];
                for (int v = 0; v < NV; v++)
                {
                    memset(y, 0, 16 * n);
                    KT[ri].f[kind][v](x, 0, y, 0, tw, 0, c, 0, c, 0, c);
                    if (!v) memcpy(ref, y, 16 * n);
                    same[v] = !memcmp(ref, y, 16 * n);
                }
                const double t0 = now_ns();
                for (int i = 0; i < 64; i++) KT[ri].f[kind][0](x, 0, y, 0, tw, 0, c, 0, c, 0, c);
                int reps = (int)(80e3 / ((now_ns() - t0) / 64));
                if (reps < 8) reps = 8;
                double t[NV * 2][21], med[NV * 2];
                for (int r = 0; r < 21; r++)
                    for (int k = 0; k < NV * 2; k++)
                    {
                        const int u = (r & 1) ? NV * 2 - 1 - k : k;
                        t[u][r] = shifted(16 * (u & 1), KT[ri].f[kind][u >> 1], x, y, tw, c, reps);
                    }
                for (int u = 0; u < NV * 2; u++)
                {
                    qsort(t[u], 21, sizeof(double), cmpd);
                    med[u] = t[u][10];
                }
                fprintf(lg, "%s R=%-3d c=%-3d", kind ? "t2" : "n1", R, c);
                const double bw = fmax(med[0], med[1]), bb = fmin(med[0], med[1]);
                for (int v = 0; v < NV; v++)
                {
                    const double a = med[2 * v], b = med[2 * v + 1], w = fmax(a, b), be = fmin(a, b);
                    fprintf(lg, " | %s %7.1f %7.1f x%.2f%s", VN[v], a, b, w / be, same[v] ? "" : " NOTBIT");
                    sum_worst[v][ci] += log(w / bw);
                    sum_best[v][ci] += log(be / bb);
                }
                fprintf(lg, "\n");
                fflush(lg);
                ncell[ci]++;
                free(ref);
                _aligned_free(x);
                _aligned_free(y);
                if (tw) _aligned_free((void *)tw);
                Sleep(200);
            }
    fprintf(lg, "\ngeomean over radices x kinds, worst state / base worst state (best state / base best state):\n");
    for (int ci = 0; ci < 3; ci++)
    {
        fprintf(lg, "c=%-3d", CNT[ci]);
        for (int v = 1; v < NV; v++)
            fprintf(lg, " | %s %.3f (%.3f)", VN[v], exp(sum_worst[v][ci] / ncell[ci]), exp(sum_best[v][ci] / ncell[ci]));
        fprintf(lg, "\n");
    }
    fclose(lg);
    printf("done\n");
    return 0;
}
