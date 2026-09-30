/* real_door_check.c -- the real door with ZTT-r, the real mono and the real
 * flat DIT wired in: at each N, both directions and both placements, the
 * door's pick (raced on a scratch store under recalibrate=1, then REPLAYED
 * from its bank) against the incumbent pinned by VFFT_ZRP=0 VFFT_ZRM=0
 * VFFT_ZRF=0 (zr2c at even N, the odd-real routes at odd N -- out of place
 * always: in place the odd incumbent runs through a copy), gated (relerr <
 * 1e-12), the roundtrip c2r(r2c(x)) = N x checked, the replay checked, then
 * the pick raced against the incumbent (the library body, 9 rounds, paced,
 * alternated).
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/real_door_check.c
 * Run:   real_door_check <scratch wisdom dir> [N | tiny | big | odd]   (the store is WRITTEN) */
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

static vfft_plan mk(vfft_wisdom *W, int N, int c2r, int ip, int recal)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1; cfg.recalibrate = recal;
    return vfft_create(&cfg);
}
/* run a plan on `in`, the result into `out` (in place: through the buffer b) */
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
    if (argc < 2) { fprintf(stderr, "usage: real_door_check <scratch wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const char *sel = argc > 2 ? argv[2] : "";
    const int only = atoi(sel);
    const int tiny_only = !strcmp(sel, "tiny"), big_only = !strcmp(sel, "big"), odd_only = !strcmp(sel, "odd");
    /* the real mono's band (every rn1 radix), then pow2, then the 2^a*odd band (ZTT-r
     * needs 2^7 * odd: the ingest's runs in whole blocks, the terminator's run length a
     * multiple of 8) */
    static const int Ns[] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27,
                              29, 31, 32, 37, 41, 43, 47, 64,
                              512, 1024, 2048, 4096, 8192, 32768,
                              384, 768, 1536, 3072, 6144, 12288, 640, 1280, 2560, 5120, 896, 1792, 3584,
                              1152, 2304, 4608, 1920, 3840, 7680, 15360,
                              /* the odd band: the real flat DIT's cells (il/real/zrf.h) */
                              33, 35, 39, 45, 49, 51, 55, 63, 75, 99, 105, 135, 165, 225, 243, 315, 405, 525, 625,
                              675, 729, 945, 1125, 1215, 1365, 1375, 1575, 2025, 2187, 3125, 3375, 6561, 10125, 16875 };
    unsigned seed = 0x777u;
    int fails = 0;
    for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
    {
        const int N = Ns[ni];
        const size_t NX = (size_t)N + 2;
        if (only > 0 && N != only) continue;
        if (tiny_only && N > 64) continue;
        if (big_only && (N <= 64 || (N & 1))) continue;
        if (odd_only && (!(N & 1) || N < 33)) continue;
        double *x = (double *)_aligned_malloc(NX * sizeof(double), 64);
        double *X0 = (double *)_aligned_malloc(NX * sizeof(double), 64), *X1 = (double *)_aligned_malloc(NX * sizeof(double), 64), *X2 = (double *)_aligned_malloc(NX * sizeof(double), 64);
        double *y0 = (double *)_aligned_malloc(NX * sizeof(double), 64), *y1 = (double *)_aligned_malloc(NX * sizeof(double), 64), *y2 = (double *)_aligned_malloc(NX * sizeof(double), 64);
        double *b = (double *)_aligned_malloc(NX * sizeof(double), 64);
        for (size_t i = 0; i < NX; i++) x[i] = urand(&seed);
        x[N] = x[N + 1] = 0;
        memset(X0, 0, NX * sizeof(double)); memset(X1, 0, NX * sizeof(double)); memset(X2, 0, NX * sizeof(double));
        for (int ip = 0; ip < ((N & 1) && N < 33 ? 1 : 2); ip++)
        {
            const int ip0 = (N & 1) ? 0 : ip;   /* the odd incumbent is out of place */
            /* the incumbent, pinned; the door's pick raced (recalibrate) then replayed */
            _putenv("VFFT_ZRP=0"); _putenv("VFFT_ZRM=0"); _putenv("VFFT_ZRF=0");
            vfft_plan f0 = mk(W, N, 0, ip0, 0), b0 = mk(W, N, 1, ip0, 0);
            _putenv("VFFT_ZRP="); _putenv("VFFT_ZRM="); _putenv("VFFT_ZRF=");
            vfft_plan f1 = mk(W, N, 0, ip, 1), b1 = mk(W, N, 1, ip, 1);
            vfft_plan f2 = mk(W, N, 0, ip, 0), b2 = mk(W, N, 1, ip, 0);
            if (!f0 || !b0 || !f1 || !b1 || !f2 || !b2) { printf("N=%d ip=%d: a plan is NULL\n", N, ip); fails++; continue; }
            run(f0, VFFT_FORWARD, ip0, x, X0, b, NX, NX);
            run(f1, VFFT_FORWARD, ip, x, X1, b, NX, NX);
            run(f2, VFFT_FORWARD, ip, x, X2, b, NX, NX);
            run(b0, VFFT_BACKWARD, ip0, X0, y0, b, NX, (size_t)N);
            run(b1, VFFT_BACKWARD, ip, X0, y1, b, NX, (size_t)N);
            run(b2, VFFT_BACKWARD, ip, X0, y2, b, NX, (size_t)N);
            const double gf = maxrel(X1, X0, NX), gb = maxrel(y1, y0, (size_t)N);
            const double g2 = maxrel(X2, X1, NX) + maxrel(y2, y1, (size_t)N);
            double rt = 0, sc = 0;
            for (int i = 0; i < N; i++) { double d = fabs(y1[i] - (double)N * x[i]); if (d > rt) rt = d; if (fabs((double)N * x[i]) > sc) sc = fabs((double)N * x[i]); }
            rt = sc > 0 ? rt / sc : rt;
            const int ok = gf < 1e-12 && gb < 1e-12 && rt < 1e-12 && g2 < 1e-12;   /* the replay: the bank builds an equivalent plan (zr2c's child re-races under recalibrate, so not bitwise) */
            if (!ok) fails++;
            printf("N=%-6d %s  vs incumbent: r2c %.1e  c2r %.1e | roundtrip %.1e | replay %.1e  %s\n",
                   N, ip ? "IP " : "OOP", gf, gb, rt, g2, ok ? "PASS" : "FAIL");
            {
                arm_t af0 = { f0, VFFT_FORWARD, ip0 ? b : x, ip0 ? b : X2 }, af1 = { f1, VFFT_FORWARD, ip ? b : x, ip ? b : X1 };
                arm_t ab0 = { b0, VFFT_BACKWARD, ip0 ? b : X0, ip0 ? b : y0 }, ab1 = { b1, VFFT_BACKWARD, ip ? b : X0, ip ? b : y1 };
                const vfft_race_arm_t arms[4] = { { "inc r2c", arm_run, &af0 }, { "door r2c", arm_run, &af1 },
                                                  { "inc c2r", arm_run, &ab0 }, { "door c2r", arm_run, &ab1 } };
                double t0 = vfft_now_ns(); arm_run(&af0); double est = vfft_now_ns() - t0;
                int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 4096) reps = 4096;
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                double ns[4];
                memcpy(b, x, NX * sizeof(double));
                vfft_race_run(&proto, arms, 4, ns);
                printf("  race: r2c incumbent %.0f door %.0f (%.2fx) | c2r incumbent %.0f door %.0f (%.2fx)  [%s | %s]\n",
                       ns[0], ns[1], ns[0] / ns[1], ns[2], ns[3], ns[2] / ns[3], vfft_plan_route(f1), vfft_plan_route(b1));
                fflush(stdout);
            }
            vfft_destroy(f0); vfft_destroy(b0); vfft_destroy(f1); vfft_destroy(b1); vfft_destroy(f2); vfft_destroy(b2);
        }
        _aligned_free(x); _aligned_free(X0); _aligned_free(X1); _aligned_free(X2); _aligned_free(y0); _aligned_free(y1); _aligned_free(y2); _aligned_free(b);
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
