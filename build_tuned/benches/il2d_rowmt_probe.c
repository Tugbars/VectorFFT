/* il2d_rowmt_probe.c — READER D S1-increment-1 experiment: does the TC row
 * door's EXISTING MT (tcbw clones + slab dispatch) light up if the row plan
 * is created with nthreads>1?  Checks (a) clone count, (b) pool state,
 * (c) bitwise equality vs the serial arm, (d) wall time per plane.
 * VFFT_IL2D_ROWMT=<n> is a temporary create-time knob patched into vfft.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)f.QuadPart;
}

static void run(int N1, int N2)
{
    const size_t RN = (size_t)N1 * N2, hp1 = N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    double *x = (double *)malloc(RN * sizeof(double));
    double *z = (double *)malloc((2 * CN + 8) * sizeof(double));
    vfft_config_t c;
    vfft_plan p;
    size_t i; int r;
    double t0, best = 1e300;

    for (i = 0; i < RN; i++) x[i] = 1.0 + 1e-6 * (double)(i & 1023);
    memset(z, 0, (2 * CN + 8) * sizeof(double));
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C; c.placement = VFFT_OUTOFPLACE;
    c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1;
    c.layout = VFFT_LAYOUT_INTERLEAVED; c.nthreads = 8; c.wisdom_write = 0;

    p = vfft_create(&c);
    if (!p) { printf("%dx%d create FAILED\n", N1, N2); return; }
    printf("%5dx%-5d ROWMT=%-3s tc_workers(rowplan) via pool=%d\n",
           N1, N2, getenv("VFFT_IL2D_ROWMT") ? getenv("VFFT_IL2D_ROWMT") : "-",
           vfft_get_num_threads());
    for (r = 0; r < 30; r++) {
        t0 = now_ns();
        vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
        { double d = now_ns() - t0; if (d < best) best = d; }
    }
    printf("            best %.1f us/plane   pool after execute = %d\n",
           best * 1e-3, vfft_get_num_threads());
    { FILE *f = fopen(getenv("VFFT_ROWMT_DUMP") ? getenv("VFFT_ROWMT_DUMP") : "rowmt.bin", "wb");
      if (f) { fwrite(z, sizeof(double), 2 * CN, f); fclose(f); } }
    vfft_destroy(p);
    free(x); free(z);
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    if (getenv("VFFT_TCMT_VERBOSE") == NULL) _putenv("VFFT_TCMT_VERBOSE=1");
    run(256, 256);
    run(1024, 1024);
    return 0;
}
