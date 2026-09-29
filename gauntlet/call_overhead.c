/* call_overhead.c -- where the nanoseconds go at tiny N through the real
 * door: per N the door's r2c and c2r, the c2c child the door wraps (c2c(N)
 * for odd N through the promote bridge, c2c(N/2) for even N through zr2c),
 * the bridge's own passes (promote + memcpy) alone, and an empty call (the
 * harness floor) -- one paced, alternated race per N, tens of thousands of
 * reps per sample. Build: python gauntlet/build.py --compile --vfft --src gauntlet/call_overhead.c
 * Run: call_overhead <wisdom dir> [N ...] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "vfft.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

typedef struct { vfft_plan h; vfft_dir_t dir; double *in, *out; int n; double *b1, *b2; } arm_t;
static void arm_exec(void *v) { arm_t *a = (arm_t *)v; vfft_execute(a->h, a->dir, a->in, NULL, a->out, NULL); }
static void arm_empty(void *v) { (void)v; }
/* the odd-real bridge's own passes: promote the reals to a complex row, copy the CCE row back */
static void arm_bridge(void *v)
{
    arm_t *a = (arm_t *)v;
    const size_t n = (size_t)a->n, hp1 = n / 2 + 1;
    for (size_t k = 0; k < n; k++) { a->b1[2 * k] = a->in[k]; a->b1[2 * k + 1] = 0.0; }
    memcpy(a->out, a->b2, 2 * hp1 * sizeof(double));
}

static vfft_plan mk(vfft_wisdom *W, int N, vfft_transform_t t)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = t; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = t == VFFT_C2C ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
    return vfft_create(&cfg);
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: call_overhead <wisdom dir> [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int Ns_def[] = { 3, 5, 7, 9, 12, 15, 16, 20, 32, 64 };
    const int nN = argc > 2 ? argc - 2 : (int)(sizeof Ns_def / sizeof Ns_def[0]);
    printf("%-5s %8s %8s %8s %8s %8s   door r2c - child - bridge = residue\n", "N", "empty", "r2c", "c2r", "child", "bridge");
    for (int i = 0; i < nN; i++)
    {
        const int N = argc > 2 ? atoi(argv[2 + i]) : Ns_def[i];
        const int Nc = (N & 1) ? N : N / 2;          /* the child's length */
        double *x = (double *)_aligned_malloc((size_t)(4 * N + 64) * sizeof(double), 64);
        double *X = (double *)_aligned_malloc((size_t)(4 * N + 64) * sizeof(double), 64);
        double *y = (double *)_aligned_malloc((size_t)(4 * N + 64) * sizeof(double), 64);
        double *b1 = (double *)_aligned_malloc((size_t)(4 * N + 64) * sizeof(double), 64);
        double *b2 = (double *)_aligned_malloc((size_t)(4 * N + 64) * sizeof(double), 64);
        for (int k = 0; k < 4 * N + 64; k++) { x[k] = 0.001 * (k % 7) - 0.003; X[k] = y[k] = b1[k] = b2[k] = 0; }
        vfft_plan hr = mk(W, N, VFFT_R2C), hb = mk(W, N, VFFT_C2R), hc = mk(W, Nc, VFFT_C2C);
        if (!hr || !hb || !hc) { printf("%-5d plans missing (r2c %p c2r %p c2c(%d) %p)\n", N, (void *)hr, (void *)hb, Nc, (void *)hc); continue; }
        vfft_execute(hr, VFFT_FORWARD, x, NULL, X, NULL);
        arm_t ar = { hr, VFFT_FORWARD, x, X, N, b1, b2 }, ab = { hb, VFFT_BACKWARD, X, y, N, b1, b2 };
        arm_t ac = { hc, VFFT_FORWARD, x, b2, N, b1, b2 }, abr = { NULL, VFFT_FORWARD, x, X, N, b1, b2 };
        const vfft_race_arm_t arms[5] = { { "empty", arm_empty, &ar }, { "r2c", arm_exec, &ar }, { "c2r", arm_exec, &ab },
                                          { "child", arm_exec, &ac }, { "bridge", arm_bridge, &abr } };
        double t0 = vfft_now_ns(); for (int r = 0; r < 1000; r++) arm_exec(&ar); double est = (vfft_now_ns() - t0) / 1000.0;
        int reps = (int)(2.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 1000) reps = 1000; if (reps > 200000) reps = 200000;
        const vfft_race_proto_t proto = { 15, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        double ns[5];
        vfft_race_run(&proto, arms, 5, ns);
        printf("%-5d %8.2f %8.2f %8.2f %8.2f %8.2f   %.2f\n", N, ns[0], ns[1], ns[2], ns[3], ns[4], ns[1] - ns[3] - ns[4]);
        fflush(stdout);
        vfft_destroy(hr); vfft_destroy(hb); vfft_destroy(hc);
        _aligned_free(x); _aligned_free(X); _aligned_free(y); _aligned_free(b1); _aligned_free(b2);
    }
    return 0;
}
