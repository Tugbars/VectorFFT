/* policy_cell_probe.c — one cell, timed where the planning policy is
 * actually called (CREATE) and where it is not (EXECUTE), so a policy
 * migration step can be checked for a latency regression.
 * (docs/design/planning_policy_design.md, 2026-09-16.)
 *
 * The cell: N (default 1048576), order NATURAL, layout INTERLEAVED, K=1,
 * both placements, T=1 — the richest single cell for step 1's laws. It
 * exercises
 *   L4 rank-1 out of place   k1_commit's `scr_req` picks the ord=nat K=1 row
 *   L4 rank-1 in place       the in-place door's `_ip_order_is_nat`
 *   L4 rank >= 2             its four-step child is a 2D cell: `il2d_ord`
 *                            and the 2D tier's eight order sites
 *   L9 the race ceiling      `_k1_il_plan_race`'s gate at a pow2 N
 *
 * The store is READ-ONLY (wisdom_write = 0) and loaded once, so every
 * create REPLAYS a banked verdict: no race, no file I/O in the timed
 * region. Pinned to core 2 at HIGH priority (the one-thread protocol).
 *
 * Run:   policy_cell_probe.exe <wisdir> [N] [reps]
 * Build: python build.py --compile --vfft --src benches/policy_cell_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : NULL;
    const int N = argc > 2 ? atoi(argv[2]) : 1048576;
    const int reps = argc > 3 ? atoi(argv[3]) : 15;
    const size_t nb = (size_t)2 * N * sizeof(double);
    double *x = (double *)_aligned_malloc(nb, 64), *y = (double *)_aligned_malloc(nb, 64);
    vfft_wisdom *W;
    int ip, r;
    if (!dir) { printf("usage: policy_cell_probe.exe <wisdir> [N] [reps]\n"); return 2; }
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x4);   /* core 2 */
    for (size_t i = 0; i < 2 * (size_t)N; i++) x[i] = (double)((i * 2654435761u) % 1000) / 1000.0;
    W = vfft_wisdom_load(dir);
    if (!W) { printf("wisdom load failed: %s\n", dir); return 2; }
    vfft_set_num_threads(1);
    printf("cell N=%d order=NATURAL layout=IL K=1 T=1, store %s, %d reps\n", N, dir, reps);

    for (ip = 0; ip < 2; ip++)
    {
        double cre[64], exe[64], first = 0;
        vfft_config_t cfg;
        vfft_plan p;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.order = VFFT_ORDER_NATURAL; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 0;   /* replay only, never bank */

        /* CREATE: the policy path. The first create in the process carries
         * the process-lifetime memos' cold cost and is reported apart. */
        {
            const double t0 = now_ns();
            p = vfft_create(&cfg);
            first = now_ns() - t0;
        }
        if (!p) { printf("  %s: create REFUSED\n", ip ? "ip " : "oop"); continue; }
        vfft_destroy(p);
        for (r = 0; r < reps && r < 64; r++)
        {
            const double t0 = now_ns();
            p = vfft_create(&cfg);
            cre[r] = now_ns() - t0;
            if (!p) { printf("  create refused at rep %d\n", r); return 1; }
            if (r < reps - 1) vfft_destroy(p);
        }
        /* EXECUTE: untouched by the policy step; the control arm */
        for (r = 0; r < 3; r++)
        {
            if (ip) { memcpy(y, x, nb); vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL); }
            else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
        }
        for (r = 0; r < reps && r < 64; r++)
        {
            double t0;
            if (ip) memcpy(y, x, nb);
            t0 = now_ns();
            if (ip) vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL);
            else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL);
            exe[r] = now_ns() - t0;
        }
        vfft_destroy(p);
        qsort(cre, (size_t)reps, sizeof cre[0], cmpd);
        qsort(exe, (size_t)reps, sizeof exe[0], cmpd);
        printf("  %s create: first %9.0f  min %8.0f  med %8.0f  max %8.0f ns  (spread %.1f%%)\n",
               ip ? "ip " : "oop", first, cre[0], cre[reps / 2], cre[reps - 1],
               100.0 * (cre[reps - 1] - cre[0]) / (cre[0] > 0 ? cre[0] : 1));
        printf("  %s exec  : min %10.0f  med %10.0f  max %10.0f ns  (spread %.1f%%)\n",
               ip ? "ip " : "oop", exe[0], exe[reps / 2], exe[reps - 1],
               100.0 * (exe[reps - 1] - exe[0]) / (exe[0] > 0 ? exe[0] : 1));
    }
    vfft_wisdom_free(W);
    _aligned_free(x); _aligned_free(y);
    return 0;
}
