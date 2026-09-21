/* chain3_skew_probe.c — does the chain3 engine's speed depend on the PAGE
 * OFFSETS of its two staging planes and the caller's buffers? (2026-09-21)
 *
 * The gauntlet benched the same banked chain3 plan 1.1-1.5x slower than the
 * planner had raced it (chain3 only; flat/pair/prime agree with the race),
 * and 47 chain3 cells ran slower when timed AFTER MKL than before it — the
 * signature of buffer PLACEMENT, not of the plan. The candidate mechanism:
 * mid1, mid2 and the caller's zin/zout come from 64-B-aligned heap blocks
 * whose 4 KB page offsets depend on the process's allocation history; when
 * a stage's read stream and its write stream sit on the same page offset
 * the core's 4K-aliasing check (a load whose low 12 address bits match an
 * in-flight store's) stalls the pipeline.
 *
 * This probe builds ONE plan for the chain given, then times the forward
 * execute with every buffer placed at a chosen page offset inside one
 * page-aligned arena, interleaving the configurations (A B C ... A B C ...)
 * so drift cancels. Nothing is banked; nothing is a verdict — this is the
 * mechanism test that precedes the fix in vfft_il3p_create.
 *
 * Run:   chain3_skew_probe.exe N R2 A B [rounds]   (chain = R2.A.B, R1 = A*B)
 * Build: python build.py --compile --vfft --src benches/chain3_skew_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "dp_planner_il.h"

typedef struct { const char *name; size_t in, mid1, mid2, out; } cfg_t;

static double now_ns(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s N R2 A B [rounds]\n", argv[0]); return 2; }
    const int N = atoi(argv[1]), R2 = atoi(argv[2]), A = atoi(argv[3]), B = atoi(argv[4]);
    const int rounds = argc > 5 ? atoi(argv[5]) : 20;
    vfft_il3p_plan_t *p = vfft_il3p_create(N, R2, A, B);
    if (!p) { fprintf(stderr, "chain %d.%d.%d does not build at N=%d\n", R2, A, B, N); return 1; }
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    const size_t bytes = (size_t)N * 2u * sizeof(double);
    const size_t slot = ((bytes + 8191u) / 4096u) * 4096u;   /* room for any offset */
    unsigned char *arena = (unsigned char *)VirtualAlloc(NULL, 4 * slot + 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!arena) return 1;
    /* page offsets (mod 4096) of zin, mid1, mid2, zout */
    static const cfg_t CFG[] = {
        { "all @0 (same page offset)",           0,    0,    0,    0 },
        { "mids @1088/@3200, caller @0 (fix)",   0, 1088, 3200,    0 },
        { "mid1==mid2 @0, caller @1088/@3200", 1088,    0,    0, 3200 },
        { "mid1 @0 mid2 @2048, caller @0",       0,    0, 2048,    0 },
        { "all distinct: 0/1088/2176/3264",      0, 1088, 2176, 3264 },
        { "caller @64 (heap-like), mids @0",    64,    0,    0,   64 },
    };
    const int ncfg = (int)(sizeof CFG / sizeof CFG[0]);
    double best[16], med[16], samples[16][64];
    double *orig1 = p->mid1, *orig2 = p->mid2;
    const int reps = N <= 512 ? 2000 : N <= 2048 ? 400 : 100;
    for (int c = 0; c < ncfg; c++) best[c] = 1e18;
    for (int r = 0; r < rounds; r++)
    {
        for (int ci = 0; ci < ncfg; ci++)
        {
            const int c = (r & 1) ? ncfg - 1 - ci : ci;           /* alternate the order */
            double *zin  = (double *)(arena + 0 * slot + CFG[c].in);
            double *mid1 = (double *)(arena + 1 * slot + CFG[c].mid1);
            double *mid2 = (double *)(arena + 2 * slot + CFG[c].mid2);
            double *zout = (double *)(arena + 3 * slot + CFG[c].out);
            p->mid1 = mid1; p->mid2 = mid2;
            for (int i = 0; i < 2 * N; i++) zin[i] = 1.0 + 1e-6 * (double)(i & 255);
            for (int w = 0; w < 10; w++) vfft_il3p_execute_fwd(p, zin, zout);
            double t0 = now_ns();
            for (int i = 0; i < reps; i++) vfft_il3p_execute_fwd(p, zin, zout);
            double t = (now_ns() - t0) / reps;
            samples[c][r < 64 ? r : 63] = t;
            if (t < best[c]) best[c] = t;
            Sleep(20);
        }
    }
    p->mid1 = orig1; p->mid2 = orig2;
    for (int c = 0; c < ncfg; c++)
    {   /* median of the rounds */
        double s[64]; int n = rounds < 64 ? rounds : 64;
        memcpy(s, samples[c], (size_t)n * sizeof(double));
        for (int i = 1; i < n; i++) for (int j = i; j > 0 && s[j - 1] > s[j]; j--) { double x = s[j]; s[j] = s[j - 1]; s[j - 1] = x; }
        med[c] = s[n / 2];
    }
    printf("chain3 N=%d chain %d.%d.%d (R1=%d): %d rounds x %d reps, pinned core 2, HIGH; page offsets of zin/mid1/mid2/zout\n",
           N, R2, A, B, A * B, rounds, reps);
    printf("  %-40s %9s %9s %7s\n", "placement", "best ns", "median", "vs fix");
    for (int c = 0; c < ncfg; c++)
        printf("  %-40s %9.0f %9.0f %6.2fx\n", CFG[c].name, best[c], med[c], med[c] / med[1]);
    VirtualFree(arena, 0, MEM_RELEASE);
    vfft_il3p_destroy(p);
    return 0;
}
