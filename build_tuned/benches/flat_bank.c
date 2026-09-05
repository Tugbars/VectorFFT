/* flat_bank.c — bank K=1 interleaved OOP verdicts for a list of N into a
 * wisdom dir through the FRONT DOOR (the K=1 plan race: pairs, chain3, flat
 * chains x forms), so the canonical bench can run those cells against MKL
 * from that store. A scratch-store tool, never the shipped store.
 *   flat_bank.exe --wisdir <dir> N [N ...]
 * Build: python build_tuned/build.py --compile --src build_tuned/benches/flat_bank.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int a0 = 1;
    if (argc >= 3 && !strcmp(argv[1], "--wisdir")) { wisdir = argv[2]; a0 = 3; }
    if (!wisdir || a0 >= argc) { printf("usage: %s --wisdir <dir> N [N ...]\n", argv[0]); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("wisdom load FAILED\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int a = a0; a < argc; a++)
    {
        const int N = atoi(argv[a]);
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.order = VFFT_ORDER_DEFAULT; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
        vfft_plan p = vfft_create(&cfg);
        printf("N=%d %s\n", N, p ? "banked" : "NO PLAN");
        if (p) vfft_destroy(p);
    }
    return 0;
}
