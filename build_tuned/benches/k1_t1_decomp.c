/* k1_t1_decomp.c — decompose the K=1 four-step's t1 stage cost: twiddle
 * (loads + cmuls) vs pure combine compute.
 *
 * Method: time radixR1_t1_oop (twiddled combine, streams tw[(l-1)*me+b]) vs
 * radixR1_n1_oop (the IDENTICAL R1-point DFT, zero twiddle loads/muls) at the
 * same (R1, me) shapes, both OOP d->d2. delta = total twiddle cost per
 * execute; table KB = (R1-1)*me*16.
 *   - If delta ~ small fraction: t1 is COMPUTE-bound -> fix = smaller R1
 *     (3-level split) / mono tier, NOT table compression.
 *   - If delta ~ half: table/cmul compression (factored or recurrence
 *     twiddles) pays directly.
 *
 * Build: python build.py --src benches/k1_t1_decomp.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

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
static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    /* (R1, R2=me) shapes from the real pair table */
    struct { int R1, R2; } S[] = {
        {64, 16}, {32, 32}, {16, 64}, {64, 32}, {32, 64}, {64, 64}, {64, 128}
    };
    int nS = (int)(sizeof(S) / sizeof(S[0]));

    printf("# t1 (twiddled combine) vs n1 (same DFT, no twiddles), OOP d->d2, hot best-of-5\n");
    printf("%-10s %8s %10s %10s %10s %8s\n", "R1 x me", "N", "t1(ns)", "n1(ns)", "tw-delta", "tblKB");

    for (int i = 0; i < nS; i++) {
        int R1 = S[i].R1, R2 = S[i].R2, N = R1 * R2;
        vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
        vfft_oop11_fn n1 = vfft_oop_leaf_fn(R1);   /* R1-point DFT leaf */
        if (!p || !n1) { printf("%4dx%-5d skip (plan/n1 missing)\n", R1, R2); continue; }

        double *dr = ad(N), *di = ad(N), *er = ad(N), *ei = ad(N);
        srand(9 + N);
        for (int n = 0; n < N; n++) {
            dr[n] = (double)rand() / RAND_MAX - 0.5;
            di[n] = (double)rand() / RAND_MAX - 0.5;
        }

        int reps = (int)(2e6 / (double)N);
        if (reps < 200) reps = 200;
        if (reps > 200000) reps = 200000;

        double bt1 = 1e18, bn1 = 1e18;
        for (int t = 0; t < 5; t++) {
            if (t) cachebust();
            /* order flip per trial */
            for (int k = 0; k < 2; k++) {
                int a = (t & 1) ? 1 - k : k;
                if (a == 0) {
                    for (int w = 0; w < 10; w++)
                        p->t1p(dr, di, er, ei, p->Qr, p->Qi, (size_t)R2, 1, (size_t)R2, 1, (size_t)R2);
                    double t0 = now_ms();
                    for (int r = 0; r < reps; r++)
                        p->t1p(dr, di, er, ei, p->Qr, p->Qi, (size_t)R2, 1, (size_t)R2, 1, (size_t)R2);
                    double ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bt1) bt1 = ns;
                } else {
                    /* n1 with the SAME memory geometry as the t1 combine:
                     * legs at stride me, groups contiguous, count=me */
                    for (int w = 0; w < 10; w++)
                        n1(dr, di, er, ei, 0, 0, (size_t)R2, 1, (size_t)R2, 1, (size_t)R2);
                    double t0 = now_ms();
                    for (int r = 0; r < reps; r++)
                        n1(dr, di, er, ei, 0, 0, (size_t)R2, 1, (size_t)R2, 1, (size_t)R2);
                    double ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bn1) bn1 = ns;
                }
            }
        }
        printf("%4dx%-5d %8d %10.0f %10.0f %10.0f %8.0f\n",
               R1, R2, N, bt1, bn1, bt1 - bn1, (double)(R1 - 1) * R2 * 16.0 / 1024.0);

        vfft_proto_aligned_free(dr); vfft_proto_aligned_free(di);
        vfft_proto_aligned_free(er); vfft_proto_aligned_free(ei);
        vfft_oop_plan_destroy(p);
    }
    printf("\nDONE\n");
    return 0;
}
