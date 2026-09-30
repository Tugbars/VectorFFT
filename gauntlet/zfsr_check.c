/* zfsr_check.c -- the real four-step (il/real/zfsr.h) against zr2c through
 * the door: per N and direction, every split of N/2 pinned by
 * VFFT_ZFSR=N1xN2 against the zr2c engine pinned by VFFT_ZRP=0, out of place
 * and in place: the output gated (relerr vs zr2c), the roundtrip
 * c2r(r2c(x)) = N x, then the arms raced (the library body, paced,
 * alternated).
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zfsr_check.c
 * Run:   zfsr_check <scratch wisdom dir> [N ...]   (default 2^20 2^21) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "vfft.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
static double maxrel(const double *a, const double *b, size_t n)
{
    double sc = 0, e = 0;
    for (size_t i = 0; i < n; i++) { if (fabs(b[i]) > sc) sc = fabs(b[i]); double d = fabs(a[i] - b[i]); if (d > e) e = d; }
    return sc > 0 ? e / sc : e;
}
typedef struct { vfft_plan h; vfft_dir_t dir; double *in, *out; } arm_t;
static void arm_run(void *v) { arm_t *a = (arm_t *)v; vfft_execute(a->h, a->dir, a->in, NULL, a->out, NULL); }

static vfft_plan mk(vfft_wisdom *W, int N, int c2r, int ip)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
    return vfft_create(&cfg);
}
static void run(vfft_plan h, vfft_dir_t dir, int ip, const double *in, double *out, double *b, size_t nin, size_t nout)
{
    if (ip) { memcpy(b, in, nin * sizeof(double)); vfft_execute(h, dir, b, NULL, b, NULL); memcpy(out, b, nout * sizeof(double)); }
    else vfft_execute(h, dir, (double *)in, NULL, out, NULL);
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zfsr_check <scratch wisdom dir> [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int sides[] = { 256, 512, 1024, 2048, 4096 };
    int Ns[16], nn = 0, fails = 0;
    unsigned seed = 0x4242u;
    for (int i = 2; i < argc && nn < 16; i++) Ns[nn++] = atoi(argv[i]);
    if (!nn) { Ns[nn++] = 1 << 20; Ns[nn++] = 1 << 21; }
    for (int ni = 0; ni < nn; ni++)
    {
        const int N = Ns[ni], M = N / 2;
        const size_t NX = (size_t)N + 2;
        double *x = (double *)_aligned_malloc(NX * sizeof(double), 64), *X0 = (double *)_aligned_malloc(NX * sizeof(double), 64);
        double *X1 = (double *)_aligned_malloc(NX * sizeof(double), 64), *y0 = (double *)_aligned_malloc(NX * sizeof(double), 64);
        double *y1 = (double *)_aligned_malloc(NX * sizeof(double), 64), *b = (double *)_aligned_malloc(NX * sizeof(double), 64);
        for (size_t i = 0; i < NX; i++) x[i] = urand(&seed);
        x[N] = x[N + 1] = 0;
        for (int ip = 0; ip < 2; ip++)
        {
            _putenv("VFFT_ZFSR="); _putenv("VFFT_ZRP=0");
            vfft_plan f0 = mk(W, N, 0, ip), b0 = mk(W, N, 1, ip);
            _putenv("VFFT_ZRP=");
            if (!f0 || !b0) { printf("N=%d ip=%d: zr2c plan NULL\n", N, ip); fails++; continue; }
            run(f0, VFFT_FORWARD, ip, x, X0, b, NX, NX);
            run(b0, VFFT_BACKWARD, ip, X0, y0, b, NX, (size_t)N);
            for (int si = 0; si < 5; si++)
            {
                const int n1 = sides[si], n2 = M / n1;
                char env[64];
                int ok2 = 0;
                if (M % n1) continue;
                for (int q = 0; q < 5; q++) if (sides[q] == n2) ok2 = 1;
                if (!ok2) continue;
                snprintf(env, sizeof env, "VFFT_ZFSR=%dx%d", n1, n2);
                _putenv(env);
                vfft_plan f1 = mk(W, N, 0, ip), b1 = mk(W, N, 1, ip);
                _putenv("VFFT_ZFSR=");
                if (!f1 || !b1) { printf("N=%d %s split %dx%d: plan NULL\n", N, ip ? "IP " : "OOP", n1, n2); fails++; continue; }
                run(f1, VFFT_FORWARD, ip, x, X1, b, NX, NX);
                run(b1, VFFT_BACKWARD, ip, X0, y1, b, NX, (size_t)N);
                const double gf = maxrel(X1, X0, NX), gb = maxrel(y1, y0, (size_t)N);
                double rt = 0, sc = 0;
                for (int i = 0; i < N; i++) { double d = fabs(y1[i] - (double)N * x[i]); if (d > rt) rt = d; if (fabs((double)N * x[i]) > sc) sc = fabs((double)N * x[i]); }
                rt = sc > 0 ? rt / sc : rt;
                const int ok = gf < 1e-11 && gb < 1e-11 && rt < 1e-11;
                if (!ok) fails++;
                {
                    arm_t af0 = { f0, VFFT_FORWARD, ip ? b : x, ip ? b : X0 }, af1 = { f1, VFFT_FORWARD, ip ? b : x, ip ? b : X1 };
                    arm_t ab0 = { b0, VFFT_BACKWARD, ip ? b : X0, ip ? b : y0 }, ab1 = { b1, VFFT_BACKWARD, ip ? b : X0, ip ? b : y1 };
                    const vfft_race_arm_t arms[4] = { { "zr2c r2c", arm_run, &af0 }, { "zfsr r2c", arm_run, &af1 },
                                                      { "zr2c c2r", arm_run, &ab0 }, { "zfsr c2r", arm_run, &ab1 } };
                    const vfft_race_proto_t proto = { 9, 1, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                    double ns[4];
                    memcpy(b, x, NX * sizeof(double));
                    vfft_race_run(&proto, arms, 4, ns);
                    printf("N=%-8d %s split %4dx%-4d  vs zr2c: r2c %.1e c2r %.1e roundtrip %.1e %s | r2c %.0f -> %.0f us (%.2fx)  c2r %.0f -> %.0f us (%.2fx)\n",
                           N, ip ? "IP " : "OOP", n1, n2, gf, gb, rt, ok ? "PASS" : "FAIL",
                           ns[0] / 1e3, ns[1] / 1e3, ns[0] / ns[1], ns[2] / 1e3, ns[3] / 1e3, ns[2] / ns[3]);
                    fflush(stdout);
                }
                vfft_destroy(f1); vfft_destroy(b1);
            }
            vfft_destroy(f0); vfft_destroy(b0);
        }
        _aligned_free(x); _aligned_free(X0); _aligned_free(X1); _aligned_free(y0); _aligned_free(y1); _aligned_free(b);
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
