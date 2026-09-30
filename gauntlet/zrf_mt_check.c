/* zrf_mt_check.c -- the real flat DIT's threaded form against its serial run,
 * through the door: per odd N, direction and placement, a chain and a tile
 * budget pinned by VFFT_ZRF=chain/t/w<tile> at one thread (the reference)
 * and at T threads with /m (the threaded form): the outputs compared BITWISE
 * (the threaded walk is a loop restriction of the serial one), the
 * engagement counter checked, then serial and threaded timed unpaced
 * (threaded arms are never paused).
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zrf_mt_check.c
 * Run:   zrf_mt_check <scratch wisdom dir> [T=8] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "vfft.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#include "sibling_guard.h"
#ifdef _WIN32
#include <windows.h>
#endif
long vfft_zrf_mt_passes(void);

typedef struct { vfft_plan h; vfft_dir_t dir; double *in, *out; } arm_t;
static void arm_run(void *v) { arm_t *a = (arm_t *)v; vfft_execute(a->h, a->dir, a->in, NULL, a->out, NULL); }

static vfft_plan mk(vfft_wisdom *W, int N, int c2r, int ip, int T)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = T;
    return vfft_create(&cfg);
}
static void run(vfft_plan h, vfft_dir_t dir, int ip, const double *in, double *out, double *b, size_t nin, size_t nout)
{
    if (ip) { memcpy(b, in, nin * sizeof(double)); vfft_execute(h, dir, b, NULL, b, NULL); memcpy(out, b, nout * sizeof(double)); }
    else vfft_execute(h, dir, (double *)in, NULL, out, NULL);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: zrf_mt_check <scratch wisdom dir> [T]\n"); return 2; }
    const int T = argc > 2 ? atoi(argv[2]) : 8;
    bench_pin_pcores();
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    vfft_set_num_threads(T);
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    /* cell, chain/t/w<tile> */
    static const struct { int N; const char *ct; } cells[] = {
        { 2025, "9.5.9.5/t/w64" }, { 6561, "9.9.9.9/t/w1024" }, { 10125, "9.5.5.9.5/t/w256" }, { 16875, "9.3.5.5.5.5/t/w1024" },
        { 50625, "9.5.5.5.9.5/t/w1024" }, { 59049, "9.9.9.9.9/t/w1024" }, { 151875, "9.5.5.5.9.5.3/t/w1024" },
        { 253125, "9.5.5.5.9.5.5/t/w1024" }, { 531441, "9.9.9.9.9.9/t/w1024" }, { 1265625, "9.5.5.5.9.5.5.5/t/w1024" },
        { 78125, "5.5.5.5.5.5.5/t/w1024" }, { 117649, "7.7.7.7.7.7/t/w512" } };
    int fails = 0;
    unsigned seed = 0x99u;
    for (int ci = 0; ci < (int)(sizeof cells / sizeof cells[0]); ci++)
    {
        const int N = cells[ci].N;
        const size_t NX = (size_t)N + 3;
        double *x = (double *)_aligned_malloc(NX * 8, 64), *X0 = (double *)_aligned_malloc(NX * 8, 64), *X1 = (double *)_aligned_malloc(NX * 8, 64);
        double *y0 = (double *)_aligned_malloc(NX * 8, 64), *y1 = (double *)_aligned_malloc(NX * 8, 64), *b = (double *)_aligned_malloc(NX * 8, 64);
        for (size_t i = 0; i < NX; i++) { seed = seed * 1664525u + 1013904223u; x[i] = (double)(seed >> 8) / (double)(1u << 24) - 0.5; }
        for (int ip = 0; ip < 2; ip++)
        {
            char env[96];
            snprintf(env, sizeof env, "VFFT_ZRF=%s", cells[ci].ct); _putenv(env);
            vfft_plan f0 = mk(W, N, 0, ip, 1), b0 = mk(W, N, 1, ip, 1);
            snprintf(env, sizeof env, "VFFT_ZRF=%s/m", cells[ci].ct); _putenv(env);
            vfft_plan f1 = mk(W, N, 0, ip, T), b1 = mk(W, N, 1, ip, T);
            _putenv("VFFT_ZRF=");
            if (!f0 || !b0 || !f1 || !b1) { printf("N=%d %s: a pin does not build (%s)\n", N, ip ? "IP " : "OOP", cells[ci].ct); fails++; continue; }
            memset(X0, 0, NX * 8); memset(y0, 0, NX * 8); memset(X1, 0, NX * 8); memset(y1, 0, NX * 8);
            run(f0, VFFT_FORWARD, ip, x, X0, b, NX, NX);
            run(b0, VFFT_BACKWARD, ip, X0, y0, b, NX, (size_t)N);
            const long e0 = vfft_zrf_mt_passes();
            run(f1, VFFT_FORWARD, ip, x, X1, b, NX, NX);
            run(b1, VFFT_BACKWARD, ip, X0, y1, b, NX, (size_t)N);
            const long eng = vfft_zrf_mt_passes() - e0;
            const int same = !memcmp(X1, X0, ((size_t)N + 1) * 8) && !memcmp(y1, y0, (size_t)N * 8);
            double rt = 0, sc = 0;
            for (int i = 0; i < N; i++) { const double d = fabs(y1[i] - (double)N * x[i]); if (d > rt) rt = d; if (fabs((double)N * x[i]) > sc) sc = fabs((double)N * x[i]); }
            if (eng != 2 || !same || rt / sc > 1e-12) fails++;
            printf("N=%-8d %s %-24s engaged %ld/2, %s, roundtrip %.1e\n", N, ip ? "IP " : "OOP", cells[ci].ct, eng,
                   eng < 2 ? "declined (serial)" : same ? "BITWISE the serial run" : "*** DIFFERS ***", rt / sc);
            for (int d = 0; d < 2; d++)
            {   /* timing: serial then threaded; unpaced, two warm passes */
                arm_t am[2] = { { d ? b0 : f0, d ? VFFT_BACKWARD : VFFT_FORWARD, ip ? b : (d ? X0 : x), ip ? b : (d ? y1 : X1) },
                                { d ? b1 : f1, d ? VFFT_BACKWARD : VFFT_FORWARD, ip ? b : (d ? X0 : x), ip ? b : (d ? y1 : X1) } };
                const vfft_race_arm_t arms[2] = { { "serial", arm_run, &am[0] }, { "threaded", arm_run, &am[1] } };
                double ns[2];
                memcpy(b, d ? X0 : x, NX * 8);
                const double t0 = vfft_now_ns(); arm_run(&am[0]); const double est = vfft_now_ns() - t0;
                int reps = (int)(2.0e6 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 4096) reps = 4096;
                const vfft_race_proto_t proto = { 7, reps, VFFT_RACE_MEDIAN, 1, 2, NULL, NULL, 0 };
                vfft_race_run(&proto, arms, 2, ns);
                printf("   %s %s: serial %.1f us  threaded %.1f us (%.2fx)\n", d ? "c2r" : "r2c", ip ? "IP " : "OOP", ns[0] / 1e3, ns[1] / 1e3, ns[0] / ns[1]);
            }
            fflush(stdout);
            vfft_destroy(f0); vfft_destroy(b0); vfft_destroy(f1); vfft_destroy(b1);
        }
        _aligned_free(x); _aligned_free(X0); _aligned_free(X1); _aligned_free(y0); _aligned_free(y1); _aligned_free(b);
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
