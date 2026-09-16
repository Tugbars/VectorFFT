/* il2d_mt_probe.c — one 2D interleaved c2c cell (scrambled, out of place:
 * the four-step's child) through the front door at T, best-of forward,
 * beside MKL DFTI 2D (natural, NOT_INPLACE) at the same T. A probe for the
 * large-plane threading question (2026-09-15); pins: VFFT_IL2D_CHAIN,
 * VFFT_IL2D_WL on a scratch store (a cold cell races under the pins).
 *
 * Run:   il2d_mt_probe.exe <wisdir> <N1> <N2> <T> [mkl=1]
 * Build: python build.py --compile --mkl --vfft --src benches/il2d_mt_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#include "mkl.h"
static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
int main(int argc, char **argv)
{
    const char *dir = argv[1];
    const int N1 = atoi(argv[2]), N2 = atoi(argv[3]), T = atoi(argv[4]);
    const int want_mkl = argc > 5 ? atoi(argv[5]) : 1;
    const int Kb = argc > 7 ? atoi(argv[7]) : 1;   /* howmany: >1 wakes the plane queue */
    const int ip = argc > 6 ? atoi(argv[6]) : 0;   /* 1 = in place (y seeded from x before every timed run) */
    const size_t PN = (size_t)N1 * N2 * (size_t)(argc > 7 ? atoi(argv[7]) : 1),
                 nb = 2 * PN * sizeof(double);
    double *x = (double *)_aligned_malloc(nb, 64), *y = (double *)_aligned_malloc(nb, 64);
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg; vfft_plan p;
    int r, reps;
    double best = 1e30, t0;
    for (size_t i = 0; i < 2 * PN; i++) x[i] = (double)((i * 2654435761u) % 1000) / 1000.0;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = Kb;
    cfg.order = (getenv("VFFT_PROBE_NAT") ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED); cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = T; cfg.wisdom = W; cfg.wisdom_write = 1;
    cfg.recalibrate = (getenv("VFFT_PROBE_RECAL") != NULL);  /* the caller's override */
    p = vfft_create(&cfg);
    if (!p) { printf("%dx%d T=%d: create refused\n", N1, N2, T); return 1; }
    vfft_set_num_threads(T);
#define RUN() do { if (ip) { memcpy(y, x, nb); vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL); } else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL); } while (0)
#define TIMED() do { if (ip) memcpy(y, x, nb); t0 = now_ns(); if (ip) vfft_execute(p, VFFT_FORWARD, y, NULL, y, NULL); else vfft_execute(p, VFFT_FORWARD, x, NULL, y, NULL); t0 = now_ns() - t0; } while (0)
    for (r = 0; r < 3; r++) RUN();
    TIMED();
    reps = (int)(300e6 / (t0 > 1 ? t0 : 1)); if (reps < 5) reps = 5; if (reps > 200) reps = 200;
    for (r = 0; r < reps; r++) { TIMED(); if (t0 < best) best = t0; }
    printf("%dx%d T=%d ours(%s %s) %10.0f ns", N1, N2, T, getenv("VFFT_PROBE_NAT") ? "nat" : "scr", ip ? "ip" : "oop", best);
    vfft_destroy(p);
    if (want_mkl)
    {
        DFTI_DESCRIPTOR_HANDLE h = NULL;
        MKL_LONG dims[2] = { N1, N2 };
        mkl_set_num_threads(T);
        if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims) == DFTI_NO_ERROR)
        {
            DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            if (DftiCommitDescriptor(h) == DFTI_NO_ERROR)
            {
                best = 1e30;
                for (r = 0; r < 3; r++) DftiComputeForward(h, x, y);
                for (r = 0; r < reps; r++) { t0 = now_ns(); DftiComputeForward(h, x, y); t0 = now_ns() - t0; if (t0 < best) best = t0; }
                printf("   MKL 2D(nat oop) %10.0f ns", best);
            }
            DftiFreeDescriptor(&h);
        }
    }
    printf("\n");
    vfft_wisdom_free(W);
    _aligned_free(x); _aligned_free(y);
    return 0;
}
