/* il2d_real_census.c — create ONE 2D interleaved REAL (r2c) cell on a cold
 * store with the tier's race log on, so its axis-race ARM LIST can be diffed
 * before and after a change to what the ladder admits (design R2,
 * 2026-09-17). The arms are the census; the winner is noise.
 *
 * Run:   VFFT_IL2D_LOG=1 il2d_real_census.exe <cold wisdir> <N1> <N2> 2>arms.txt
 * Build: python build.py --compile --vfft --src benches/il2d_real_census.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int N1 = argc > 2 ? atoi(argv[2]) : 64, N2 = argc > 3 ? atoi(argv[3]) : 64;
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_PATIENT;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    printf("%dx%d real IL: %s\n", N1, N2, p ? "created" : "REFUSED");
    if (p) vfft_destroy(p);
    vfft_wisdom_free(W);
    return p ? 0 : 1;
}
