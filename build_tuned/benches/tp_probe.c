/* the natural transpose's kernel variants at N1 x N2 complexes (default the
 * 4M split 1024x4096), a bit-reversed row permutation, T=1 and T=8
 * (persistent spinning threads); every variant checked bitwise against the
 * shipped kernel. A probe, not a bench. Build: python build.py --compile
 * --src benches/tp_probe.c */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int N1, N2; static int *perm; static const double *S; static double *D;
typedef void (*kern_t)(int k1lo, int k1hi);
/* A: the shipped kernel, TB x TB scalar, k1 blocks outer */
static void kA(int TB, int k1lo, int k1hi)
{
    for (int k1b = k1lo; k1b < k1hi; k1b += TB)
        for (int k2b = 0; k2b < N2; k2b += TB)
            for (int j = 0; j < TB; j++)
            {
                const int k1 = k1b + j;
                const double *row = S + 2 * ((size_t)perm[k1] * N2 + k2b);
                for (int i = 0; i < TB; i++)
                {
                    double *o = D + 2 * ((size_t)(k2b + i) * N1 + k1);
                    o[0] = row[2 * i]; o[1] = row[2 * i + 1];
                }
            }
}
static void kA16(int lo, int hi) { kA(16, lo, hi); }
static void kA32(int lo, int hi) { kA(32, lo, hi); }
static void kA64(int lo, int hi) { kA(64, lo, hi); }
/* B: 16x16 through a local block, output rows stored as whole lines (regular or streaming) */
static void kB(int nt, int k1lo, int k1hi)
{
    __attribute__((aligned(64))) double buf[16 * 16 * 2];
    for (int k1b = k1lo; k1b < k1hi; k1b += 16)
        for (int k2b = 0; k2b < N2; k2b += 16)
        {
            for (int j = 0; j < 16; j++)
            {
                const double *row = S + 2 * ((size_t)perm[k1b + j] * N2 + k2b);
                for (int i = 0; i < 16; i++) { buf[2 * (i * 16 + j)] = row[2 * i]; buf[2 * (i * 16 + j) + 1] = row[2 * i + 1]; }
            }
            for (int i = 0; i < 16; i++)
            {
                double *o = D + 2 * ((size_t)(k2b + i) * N1 + k1b);
                const double *b = buf + 2 * i * 16;
                for (int q = 0; q < 32; q += 4)
                {
                    __m256d v = _mm256_load_pd(b + q);
                    if (nt) _mm256_stream_pd(o + q, v); else _mm256_store_pd(o + q, v);
                }
            }
        }
    if (nt) _mm_sfence();
}
static void kB0(int lo, int hi) { kB(0, lo, hi); }
static void kB1(int lo, int hi) { kB(1, lo, hi); }
/* C: 2x2-complex AVX2 lane permutes, 16x16 blocks, streaming out of a local block */
static void kC_blocks(int k2outer, int k1lo, int k1hi)
{
    __attribute__((aligned(64))) double buf[16 * 16 * 2];
    const int no = k2outer ? N2 : (k1hi - k1lo), ni = k2outer ? (k1hi - k1lo) : N2;
    for (int ob = 0; ob < no; ob += 16)
        for (int ib = 0; ib < ni; ib += 16)
        {
            const int k1b = k2outer ? k1lo + ib : k1lo + ob, k2b = k2outer ? ob : ib;
            for (int j = 0; j < 16; j += 2)
            {
                const double *r0 = S + 2 * ((size_t)perm[k1b + j] * N2 + k2b);
                const double *r1 = S + 2 * ((size_t)perm[k1b + j + 1] * N2 + k2b);
                for (int i = 0; i < 16; i += 2)
                {
                    __m256d a = _mm256_loadu_pd(r0 + 2 * i), b = _mm256_loadu_pd(r1 + 2 * i);
                    _mm256_store_pd(buf + 2 * (i * 16 + j), _mm256_permute2f128_pd(a, b, 0x20));
                    _mm256_store_pd(buf + 2 * ((i + 1) * 16 + j), _mm256_permute2f128_pd(a, b, 0x31));
                }
            }
            for (int i = 0; i < 16; i++)
            {
                double *o = D + 2 * ((size_t)(k2b + i) * N1 + k1b);
                const double *b = buf + 2 * i * 16;
                for (int q = 0; q < 32; q += 4) _mm256_stream_pd(o + q, _mm256_load_pd(b + q));
            }
        }
    _mm_sfence();
}
static void kC(int lo, int hi) { kC_blocks(0, lo, hi); }
static void kD(int lo, int hi) { kC_blocks(1, lo, hi); }
/* the pool: T-1 spinning workers on a generation counter */
static volatile LONG g_gen = 0, g_done = 0; static kern_t g_fn; static int g_T;
static DWORD WINAPI worker(LPVOID v)
{
    const int t = (int)(intptr_t)v; LONG seen = 0;
    for (;;)
    {
        while (g_gen == seen) _mm_pause();
        seen = g_gen;
        if (seen < 0) return 0;
        { const int nb = N1 / 16; g_fn((int)((long)nb * t / g_T) * 16, (int)((long)nb * (t + 1) / g_T) * 16); }
        InterlockedIncrement(&g_done);
    }
}
static void run(kern_t fn, int T)
{
    g_fn = fn; g_T = T;
    if (T == 1) { fn(0, N1); return; }
    g_done = 0; InterlockedIncrement(&g_gen);
    { const int nb = N1 / 16; fn(0, (int)((long)nb / T) * 16); }
    while (g_done < T - 1) _mm_pause();
}
int main(int argc, char **argv)
{
    N1 = argc > 1 ? atoi(argv[1]) : 1024; N2 = argc > 2 ? atoi(argv[2]) : 4096;
    const size_t nb = (size_t)2 * N1 * N2 * sizeof(double);
    double *src = (double *)_aligned_malloc(nb, 64), *ref = (double *)_aligned_malloc(nb, 64);
    D = (double *)_aligned_malloc(nb, 64); S = src;
    perm = (int *)malloc(N1 * sizeof(int));
    {
        int bits = 0;
        while ((1 << bits) < N1) bits++;
        for (int k = 0; k < N1; k++) { int r = 0; for (int b = 0; b < bits; b++) if (k & (1 << b)) r |= 1 << (bits - 1 - b); perm[k] = r; }
    }
    for (size_t i = 0; i < (size_t)2 * N1 * N2; i++) src[i] = (double)(i % 977) * 0.001;
    kA16(0, N1); memcpy(ref, D, nb);
    struct { const char *name; kern_t fn; } K[] = {
        { "A16 shipped scalar", kA16 }, { "A32 scalar", kA32 }, { "A64 scalar", kA64 },
        { "B  block+store", kB0 }, { "B  block+stream", kB1 },
        { "C  avx2 perm+stream", kC }, { "D  k2-outer perm+stream", kD } };
    printf("%dx%d complexes (%.0f MB each way), best of 7\n", N1, N2, nb / 1048576.0);
    for (int T = 1; T <= 8; T += 7)
    {
        HANDLE th[8];
        if (T > 1) for (int t = 1; t < T; t++) th[t] = CreateThread(NULL, 0, worker, (LPVOID)(intptr_t)t, 0, NULL);
        for (int k = 0; k < 7; k++)
        {
            double best = 1e30;
            memset(D, 0, nb);
            run(K[k].fn, T);
            const int ok = memcmp(D, ref, nb) == 0;
            for (int r = 0; r < 7; r++) { double t0 = now_ns(); run(K[k].fn, T); double t = now_ns() - t0; if (t < best) best = t; }
            printf("  T=%d %-26s %9.0f us  %5.1f GB/s  %s\n", T, K[k].name, best / 1e3, 2.0 * nb / best, ok ? "bitwise" : "WRONG");
        }
        if (T > 1) { g_gen = -1; for (int t = 1; t < T; t++) WaitForSingleObject(th[t], INFINITE); g_gen = 0; }
    }
    return 0;
}
