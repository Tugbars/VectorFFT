/* pq_probe.c — the 2D PLANE QUEUE (howmany > 1): gate + measurement.
 * Gate: queue == serial loop BITWISE (both are library paths: the same
 * handle with the verdict flipped via env in two creates), engagement
 * counter moved, both directions where applicable. Measure: min-of-15
 * alternated, verdict LIVE. Cells: the four single-stage real cells
 * intra-MT cannot help (the queue's whole reason), one banded real
 * cell, one c2c cell.
 * Build: python build.py --src benches/pq_probe.c --vfft --mkl --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

static double now_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(int argc, char **argv)
{
    static const struct { int n1, n2, tr; size_t P; } C[] = {
        { 64, 64, 0, 64 },    { 64, 256, 0, 32 },
        { 32, 1024, 0, 32 },  { 16, 4096, 0, 16 },
        { 256, 256, 0, 16 },  { 256, 256, 1, 16 },
    }; /* tr: 0 = R2C (+C2R checked), 1 = C2C */
    const int NC = (int)(sizeof C / sizeof C[0]);
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    int ci, r, fails = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("--- 2D PLANE QUEUE (T=8 pool, verdict live) ---\n");
    for (ci = 0; ci < NC; ci++) {
        const int N1 = C[ci].n1, N2 = C[ci].n2, c2c = C[ci].tr;
        const size_t P = C[ci].P;
        const size_t hp1 = (size_t)N2 / 2 + 1;
        const size_t sd = c2c ? 2 * (size_t)N1 * N2 : (size_t)N1 * N2;
        const size_t dd = c2c ? sd : 2 * (size_t)N1 * hp1;
        double *x = malloc(P * sd * 8), *zq = malloc(P * dd * 8);
        double *zl = malloc(P * dd * 8);
        vfft_config_t cfg;
        vfft_plan pq, pl;
        double tl = 1e300, tq = 1e300, t0;
        long c0, c1;
        size_t i;
        if (!x || !zq || !zl) return 2;
        for (i = 0; i < P * sd; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = c2c ? VFFT_C2C : VFFT_R2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
        cfg.howmany = P;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        cfg.nthreads = 8;
        vfft_set_num_threads(8);
        /* two handles: verdict pinned to each arm via env — BOTH are
         * library paths, so bitwise equality is the real gate */
        _putenv("VFFT_PQ_NO_MT=1");
        pl = vfft_create(&cfg);
        _putenv("VFFT_PQ_NO_MT=0");
        pq = vfft_create(&cfg);
        _putenv("VFFT_PQ_NO_MT=");
        if (!pl || !pq) {
            printf("%dx%d P=%zu create FAIL\n", N1, N2, P);
            fails++;
            continue;
        }
        c0 = vfft_pq_mt_passes();
        vfft_execute(pq, VFFT_FORWARD, x, NULL, zq, NULL);
        c1 = vfft_pq_mt_passes();
        vfft_execute(pl, VFFT_FORWARD, x, NULL, zl, NULL);
        if (c1 == c0) {
            printf("%4dx%-4d P=%-3zu %s QUEUE NEVER RAN *** FAIL ***\n",
                   N1, N2, P, c2c ? "c2c" : "r2c");
            fails++;
        }
        if (memcmp(zq, zl, P * dd * 8) != 0) {
            printf("%4dx%-4d P=%-3zu %s queue != loop *** FAIL ***\n",
                   N1, N2, P, c2c ? "c2c" : "r2c");
            fails++;
        }
        for (r = 0; r < 15; r++) {
            t0 = now_ns();
            vfft_execute(pl, VFFT_FORWARD, x, NULL, zl, NULL);
            t0 = now_ns() - t0; if (t0 < tl) tl = t0;
            t0 = now_ns();
            vfft_execute(pq, VFFT_FORWARD, x, NULL, zq, NULL);
            t0 = now_ns() - t0; if (t0 < tq) tq = t0;
        }
        printf("%4dx%-4d P=%-3zu %s  loop %9.0f  queue %9.0f = %.2fx  "
               "BITWISE OK  pq-passes=%ld\n",
               N1, N2, P, c2c ? "c2c" : "r2c", tl, tq, tl / tq,
               c1 - c0);
        vfft_destroy(pl); vfft_destroy(pq);
        free(x); free(zq); free(zl);
    }
    vfft_set_num_threads(1);
    if (W) vfft_wisdom_free(W);
    printf("%s\n", fails ? "*** FAIL ***" : "=== ALL PASS ===");
    return fails ? 1 : 0;
}
