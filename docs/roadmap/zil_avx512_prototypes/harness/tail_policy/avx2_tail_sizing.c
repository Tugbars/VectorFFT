/* avx2_tail_sizing.c -- what the AVX2 IL odd-count tail costs today (2026-09-28).
 * At AVX2 a ymm holds 2 complex, so an odd column count leaves ONE column, which
 * runs the narrow 128-bit copy of the kernel's DAG. For odd radices (the kernels
 * of odd-N 1D plans) this times one n1 and one t2 call at every count 1..18 and
 * reports, per odd count c, the leftover column's cost t(c) - t(c-1) against the
 * per-column cost of the whole passes, (t(c+1) - t(c-1)) / 2: 1.0 = the tail is
 * as cheap as any column, 2.0 = it costs a full two-column pass.
 * Protocol: the calling thread pinned to core 2 at HIGH priority, each count the
 * minimum of 5 batches, 21 rounds with the counts in alternating order, median.
 * Windows (QPC); links the build's AVX2 codelet library.
 * Build: python gauntlet/build.py --compile --src <this file> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#define RADICES(X) X(3) X(5) X(7) X(9) X(11) X(13) X(15) X(25) X(27) X(43)
#define DN(R) void radix##R##_z_n1_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define DT(R) void radix##R##_z_t2_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
RADICES(DN) RADICES(DT)
#define EN(R) { R, radix##R##_z_n1_fwd_avx2, radix##R##_z_t2_fwd_avx2 },
static const struct { int R; kfn n1, t2; } K[] = { RADICES(EN) };
enum { CMAX = 18, ROUNDS = 21, VW = 4 };

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
/* the AVX2 VTW2 stream for count columns: group g of 2 columns, leg l: a record of
 * 2*VW doubles [c,c per column][-s,+s per column] (the tail harness's layout) */
static double *vtw2(int R, int count, int N)
{
    const int groups = (count + 1) / 2;
    double *t = _aligned_malloc(sizeof(double) * ((size_t)groups * (R - 1) * 2 * VW + 64), 64);
    for (int g = 0; g < groups; g++)
        for (int l = 1; l < R; l++)
        {
            double *rec = t + ((size_t)g * (R - 1) + (l - 1)) * 2 * VW;
            for (int j = 0; j < 2; j++)
            {
                int k = 2 * g + j;
                if (k >= count) k = count - 1;
                const double a = -2.0 * 3.14159265358979323846 * l * k / N, c = cos(a), s = sin(a);
                rec[2 * j] = rec[2 * j + 1] = c;
                rec[VW + 2 * j] = -s;
                rec[VW + 2 * j + 1] = s;
            }
        }
    return t;
}
static double time_call(kfn f, int t2, const double *x, double *y, const double *tw, int c, int reps)
{
    double best = 1e30;
    for (int b = 0; b < 5; b++)
    {
        const double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            f(x, 0, y, 0, t2 ? tw : 0, 0, (size_t)c, 0, (size_t)c, 0, (size_t)c);
        const double tt = (now_ns() - t0) / reps;
        if (tt < best) best = tt;
    }
    return best;
}

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    printf("AVX2 IL tail sizing: the leftover column's cost / an ordinary column's (1.0 = free, 2.0 = a full pass)\n");
    printf("%-6s %-3s", "kernel", "R");
    for (int c = 1; c < CMAX; c += 2) printf(" %6s", c == 1 ? "c=1" : "");
    printf("\n");
    for (size_t ki = 0; ki < sizeof K / sizeof *K; ki++)
        for (int t2 = 0; t2 < 2; t2++)
        {
            const int R = K[ki].R;
            const kfn f = t2 ? K[ki].t2 : K[ki].n1;
            const size_t n = (size_t)2 * R * CMAX;
            double *x = _aligned_malloc(8 * n + 64, 64), *y = _aligned_malloc(8 * n + 64, 64);
            double *tw[CMAX + 1], t[CMAX + 1][ROUNDS], med[CMAX + 1];
            const int reps = (int)(2e5 / R) + 20;
            for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
            for (int c = 1; c <= CMAX; c++) tw[c] = t2 ? vtw2(R, c, R * c) : NULL;
            for (int r = 0; r < ROUNDS; r++)
                for (int k = 1; k <= CMAX; k++)
                {
                    const int c = (r & 1) ? CMAX + 1 - k : k;
                    t[c][r] = time_call(f, t2, x, y, tw[c], c, reps);
                }
            for (int c = 1; c <= CMAX; c++)
            {
                qsort(t[c], ROUNDS, sizeof(double), cmpd);
                med[c] = t[c][ROUNDS / 2];
            }
            printf("%-6s %-3d", t2 ? "t2" : "n1", R);
            /* c = 1: the whole call is the tail; against half a two-column call */
            printf(" %6.2f", med[1] / (med[2] / 2.0));
            for (int c = 3; c < CMAX; c += 2)
            {
                const double tail = med[c] - med[c - 1], col = (med[c + 1] - med[c - 1]) / 2.0;
                printf(" %6.2f", col > 0 ? tail / col : 0.0);
            }
            printf("   | ns: c=2 %.0f, c=18 %.0f\n", med[2], med[CMAX]);
            fflush(stdout);
            for (int c = 1; c <= CMAX; c++) if (tw[c]) _aligned_free(tw[c]);
            _aligned_free(x); _aligned_free(y);
        }
    printf("columns: c = 1, 3, 5, ..., 17\n");
    return 0;
}
