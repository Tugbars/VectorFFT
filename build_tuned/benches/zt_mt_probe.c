/* zt_mt_probe.c — INC-Z gate + measurement: the K=1 1D c2c IL cascade
 * (zturn) ST vs MT. Gate half: MT == ST BITWISE both directions + the
 * engagement counter must move (the two-gate law). Measure half:
 * same-run alternated min-of-20 with the raced verdict LIVE.
 * Build: python build.py --src benches/zt_mt_probe.c --vfft --mkl --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(int argc, char **argv)
{
    static const int NS[] = { 2048, 4096, 8192, 16384, 65536, 262144 };
    const int NN = (int)(sizeof NS / sizeof NS[0]);
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    int ni, r, fails = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("--- INC-Z: K=1 zturn cascade MT (verdict live, T=8) ---\n");
    for (ni = 0; ni < NN; ni++) {
        const int N = NS[ni];
        double *x = malloc(2 * (size_t)N * 8);
        double *zs = malloc(2 * (size_t)N * 8), *zm = malloc(2 * (size_t)N * 8);
        double *ys = malloc(2 * (size_t)N * 8), *ym = malloc(2 * (size_t)N * 8);
        vfft_config_t cfg;
        vfft_plan ps, pm;
        double fs = 1e300, fm = 1e300, bs = 1e300, bm = 1e300, t0;
        long c0, c1;
        size_t i;
        if (!x || !zs || !zm || !ys || !ym) return 2;
        for (i = 0; i < 2 * (size_t)N; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.order = VFFT_ORDER_SCRAMBLED; /* the cascade tier's contract */
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        vfft_set_num_threads(8);
        cfg.nthreads = 1;
        ps = vfft_create(&cfg);
        cfg.nthreads = 8;
        pm = vfft_create(&cfg);
        if (!ps || !pm) { printf("N=%d create FAIL\n", N); fails++; continue; }
        /* gate: bitwise + engagement, both directions */
        c0 = vfft_zt_mt_passes();
        vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL);
        vfft_execute(pm, VFFT_BACKWARD, zm, NULL, ym, NULL);
        c1 = vfft_zt_mt_passes();
        vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL);
        vfft_execute(ps, VFFT_BACKWARD, zs, NULL, ys, NULL);
        if (memcmp(zs, zm, 2 * (size_t)N * 8) != 0 ||
            memcmp(ys, ym, 2 * (size_t)N * 8) != 0) {
            printf("N=%-7d MT != ST *** FAIL ***\n", N);
            fails++;
        }
        /* measure: alternated min-of-20 */
        for (r = 0; r < 20; r++) {
            t0 = now_ns();
            vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL);
            t0 = now_ns() - t0; if (t0 < fs) fs = t0;
            t0 = now_ns();
            vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL);
            t0 = now_ns() - t0; if (t0 < fm) fm = t0;
            t0 = now_ns();
            vfft_execute(ps, VFFT_BACKWARD, zs, NULL, ys, NULL);
            t0 = now_ns() - t0; if (t0 < bs) bs = t0;
            t0 = now_ns();
            vfft_execute(pm, VFFT_BACKWARD, zm, NULL, ym, NULL);
            t0 = now_ns() - t0; if (t0 < bm) bm = t0;
        }
        printf("N=%-7d fwd ST %8.0f MT %8.0f = %.2fx | bwd ST %8.0f MT "
               "%8.0f = %.2fx | %s zt-passes=%ld\n",
               N, fs, fm, fs / fm, bs, bm, bs / bm,
               c1 > c0 ? "ENGAGED" : "serial-verdict", c1 - c0);
        vfft_destroy(ps); vfft_destroy(pm);
        free(x); free(zs); free(zm); free(ys); free(ym);
    }
    vfft_set_num_threads(1);
    if (W) vfft_wisdom_free(W);
    printf("%s\n", fails ? "*** FAIL ***" : "=== ALL BITWISE ===");
    return fails ? 1 : 0;
}
