/* stack_align_probe.c -- does a kernel's speed depend on the caller's stack
 * alignment? The same call at stack shifts 0/16/32/48 bytes (alloca), same
 * data, hot in L1; 15 rounds alternating, min of 5 batches, median. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <windows.h>
#define KARGS const double *, const double *, double *, double *, const double *, const double *, \
              size_t, size_t, size_t, size_t, size_t
typedef void (*kfn)(KARGS);
void radix29_z_t2_fwd_avx2_narrow(KARGS);
void radix29_z_t2_fwd_avx2_blk_narrow(KARGS);
void radix43_z_t2_fwd_avx2_narrow(KARGS);
void radix43_z_n1_fwd_avx2_narrow(KARGS);
void radix23_z_t2_fwd_avx2_narrow(KARGS);
void radix13_z_n1_fwd_avx2_narrow(KARGS);
void radix7_z_t2_fwd_avx2_narrow(KARGS);
static const struct { const char *n; kfn f; int R, t2; } K[] = {
    { "r29 t2 narrow", radix29_z_t2_fwd_avx2_narrow, 29, 1 },
    { "r29 t2 blk_nar", radix29_z_t2_fwd_avx2_blk_narrow, 29, 1 },
    { "r43 t2 narrow", radix43_z_t2_fwd_avx2_narrow, 43, 1 },
    { "r43 n1 narrow", radix43_z_n1_fwd_avx2_narrow, 43, 0 },
    { "r23 t2 narrow", radix23_z_t2_fwd_avx2_narrow, 23, 1 },
    { "r13 n1 narrow", radix13_z_n1_fwd_avx2_narrow, 13, 0 },
    { "r7 t2 narrow", radix7_z_t2_fwd_avx2_narrow, 7, 1 },
};
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
static __attribute__((noinline)) double shifted(int shift, kfn f, const double *x, double *y, const double *tw, size_t c, int reps)
{
    volatile char *pad = alloca(64 + shift);
    pad[0] = 0;
    return timed(f, x, y, tw, c, reps);
}
static __attribute__((noinline)) void *sp_probe(int shift)
{
    volatile char *pad = alloca(64 + shift);
    pad[0] = 0;
    return (void *)__builtin_frame_address(0);
}
int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    const size_t c = 25;
    double *x = _aligned_malloc(16 * 47 * 26 + 128, 64), *y = _aligned_malloc(16 * 47 * 26 + 128, 64);
    double *tw = _aligned_malloc(8 * 13 * 46 * 8 + 128, 64);
    for (int i = 0; i < 2 * 47 * 26; i++) x[i] = sin(0.37 * i);
    for (int i = 0; i < 13 * 46 * 8; i++) tw[i] = cos(0.11 * i);
    printf("stack shift:         0        16        32        48   (ns per call, count %zu)\n", c);
    for (size_t k = 0; k < sizeof K / sizeof *K; k++)
    {
        double t[4][15];
        const int reps = 20000;
        for (int r = 0; r < 15; r++)
            for (int j = 0; j < 4; j++)
            {
                const int s = (r & 1) ? 3 - j : j;
                t[s][r] = shifted(16 * s, K[k].f, x, y, tw, c, reps / K[k].R * 10);
            }
        printf("%-16s", K[k].n);
        for (int s = 0; s < 4; s++)
        {
            qsort(t[s], 15, sizeof(double), cmpd);
            printf(" %9.1f", t[s][7]);
        }
        printf("\n");
        Sleep(100);
    }
    (void)sp_probe;
    return 0;
}
