/* il2d_c2c_mt_probe.c — INC-C measurement: the c2c IL 2D tier ST vs MT,
 * same process, alternated, min-of-20, both directions, engagement
 * printed (the two-gate law). The raced cmt verdict is LIVE (no env
 * pin): a cell whose race banks "serial" shows MT == ST time by design.
 * Build: python build.py --src benches/il2d_c2c_mt_probe.c --vfft --mkl --compile */
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
    static const int C[][2] = { { 256, 256 },   { 512, 512 },
                                { 1024, 1024 }, { 4096, 64 },
                                { 8192, 64 },   { 64, 1024 } };
    const int NC = (int)(sizeof C / sizeof C[0]);
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    int ci, r;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("--- INC-C c2c IL 2D MT (raced verdict live, T=8) ---\n");
    for (ci = 0; ci < NC; ci++) {
        const int N1 = C[ci][0], N2 = C[ci][1];
        const size_t PN = (size_t)N1 * N2;
        double *x = malloc(2 * PN * 8), *zs = malloc(2 * PN * 8);
        double *zm = malloc(2 * PN * 8);
        vfft_config_t cfg;
        vfft_plan ps, pm;
        double fs = 1e300, fm = 1e300, bs = 1e300, bm = 1e300, t0;
        long c0, c1;
        size_t i;
        if (!x || !zs || !zm) return 2;
        for (i = 0; i < 2 * PN; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_INPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
        cfg.howmany = 1;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        vfft_set_num_threads(8);
        cfg.nthreads = 1;
        ps = vfft_create(&cfg);
        cfg.nthreads = 8;
        pm = vfft_create(&cfg);
        if (!ps || !pm) { printf("%dx%d create FAIL\n", N1, N2); continue; }
        memcpy(zs, x, 2 * PN * 8);
        memcpy(zm, x, 2 * PN * 8);
        vfft_execute(pm, VFFT_FORWARD, zm, NULL, zm, NULL); /* warm */
        c0 = vfft_il2d_col_mt_passes();
        for (r = 0; r < 20; r++) {
            t0 = now_ns();
            vfft_execute(ps, VFFT_FORWARD, zs, NULL, zs, NULL);
            t0 = now_ns() - t0; if (t0 < fs) fs = t0;
            t0 = now_ns();
            vfft_execute(pm, VFFT_FORWARD, zm, NULL, zm, NULL);
            t0 = now_ns() - t0; if (t0 < fm) fm = t0;
            t0 = now_ns();
            vfft_execute(ps, VFFT_BACKWARD, zs, NULL, zs, NULL);
            t0 = now_ns() - t0; if (t0 < bs) bs = t0;
            t0 = now_ns();
            vfft_execute(pm, VFFT_BACKWARD, zm, NULL, zm, NULL);
            t0 = now_ns() - t0; if (t0 < bm) bm = t0;
        }
        c1 = vfft_il2d_col_mt_passes();
        printf("%5dx%-4d  fwd ST %9.0f MT %9.0f = %.2fx | bwd ST %9.0f "
               "MT %9.0f = %.2fx | colmt-passes=%ld\n",
               N1, N2, fs, fm, fs / fm, bs, bm, bs / bm, c1 - c0);
        vfft_destroy(ps); vfft_destroy(pm);
        free(x); free(zs); free(zm);
    }
    vfft_set_num_threads(1);
    if (W) vfft_wisdom_free(W);
    return 0;
}
