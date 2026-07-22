/* k1_stage_probe.c — falsification probe: spills vs cache residency at high N.
 *
 * Holds the t1 kernel FIXED (radix-64 t1_oop_ul, ~480 spills, R1=64) and
 * sweeps R2 = N/64 so N crosses the L1d boundary: N = 256..8192, t1 working
 * set = col scratch (16N B) + dst (16N B) + Q tables (16N B) ~ 48N B, i.e.
 * ~12KB at 256 (L1-resident) to ~384KB at 8192 (L2). Times each stage of the
 * 2pa route ALONE, hot, and reports ns/element.
 *
 *   spill hypothesis  -> per-element t1 cost FLAT in N (same code, same
 *                        spills; per-call overhead even predicts a mild
 *                        DECREASE as N amortizes it)
 *   cache hypothesis  -> step between N=1024 (~48KB ~ L1d) and N=2048
 *                        (~96KB), slow growth after
 *
 * Methodology: pinned core 2 P-core, HIGH prio, per-stage hot 10-warmup +
 * reps, best-of-5 trials, 32MB cachebust between trials, stage order flipped
 * per trial. Run one N per process (driver script).
 *
 * Build: python build.py --src benches/k1_stage_probe.c
 * Run:   k1_stage_probe <N>   (N divisible by 64; pair fixed at 64 x N/64)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg); (void)reg;

    int Nd[] = { 256, 512, 1024, 2048, 4096, 8192 };
    int nN = argc > 1 ? argc - 1 : 6;

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 1 ? atoi(argv[ni + 1]) : Nd[ni];
        int R1 = 64, R2 = N / 64;
        vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
        if (!p || !p->t1_ul) {
            fprintf(stderr, "no 2pa plan for N=%d (64x%d)\n", N, R2);
            continue;
        }
        double *wr = ad(N), *wi = ad(N), *dr = ad(N), *di = ad(N);
        srand(42 + N);
        for (int n = 0; n < N; n++) {
            wr[n] = (double)rand() / RAND_MAX - 0.5;
            wi[n] = (double)rand() / RAND_MAX - 0.5;
        }

        int reps = (int)(2e6 / (double)N);
        if (reps < 100) reps = 100;
        if (reps > 400000) reps = 400000;

        /* stage arms, exactly the 2pa route's two calls */
        double bl = 1e18, bt = 1e18;
        for (int t = 0; t < 5; t++) {
            if (t) cachebust();
            for (int k = 0; k < 2; k++) {
                int a = (t & 1) ? 1 - k : k;
                double t0, ns;
                if (a == 0) {   /* leaf: wr -> col scratch */
                    for (int w = 0; w < 10; w++)
                        p->leaf(wr, wi, p->col_re, p->col_im, 0, 0, R1, 1, R1, 1, R1);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++)
                        p->leaf(wr, wi, p->col_re, p->col_im, 0, 0, R1, 1, R1, 1, R1);
                    ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bl) bl = ns;
                } else {        /* t1-UL: col scratch -> dst (radix-64, fixed) */
                    for (int w = 0; w < 10; w++)
                        p->t1_ul(p->col_re, p->col_im, dr, di, p->Qr, p->Qi,
                                 1, R1, R2, 1, R2);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++)
                        p->t1_ul(p->col_re, p->col_im, dr, di, p->Qr, p->Qi,
                                 1, R1, R2, 1, R2);
                    ns = (now_ms() - t0) * 1e6 / reps;
                    if (ns < bt) bt = ns;
                }
            }
        }
        printf("STAGE,%d,64x%d,ws_kb,%.0f,leaf,%.1f,leaf_ns_per_el,%.3f,t1,%.1f,t1_ns_per_el,%.3f\n",
               N, R2, 48.0 * N / 1024.0, bl, bl / N, bt, bt / N);
        vfft_oop_plan_destroy(p);
    }
    printf("DONE\n");
    return 0;
}
