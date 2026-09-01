/* natip_probe.c — one cold NATURAL in-place create per sample on a fresh
 * bundle: fires the tape race and the ZCASC (>=2048) or ILP (<2048) race.
 * The [natorder] stderr lines are the whole result (VFFT_NAT_LOG=1).
 * usage: probe N [samples=1] [wisdir-base=natip_wis]
 * build: build.py --src benches/natip_probe.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int S = argc > 2 ? atoi(argv[2]) : 1;
    const char *base = argc > 3 ? argv[3] : "natip_wis";
    static char envbuf[] = "VFFT_NAT_LOG=1";
    putenv(envbuf);
    setvbuf(stderr, NULL, _IONBF, 0);
    for (int s = 0; s < S; s++)
    {
        char dir[512];
        if (S == 1) snprintf(dir, sizeof dir, "%s", base); /* exact dir: bank into a real store */
        else        snprintf(dir, sizeof dir, "%s_%d", base, s);
        vfft_wisdom *W = vfft_wisdom_load(dir);
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_INPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.dims = 1;
        cfg.n[0] = N;
        cfg.howmany = 1;
        cfg.rigor = VFFT_MEASURE;
        cfg.wisdom = W;
        cfg.wisdom_write = 1; /* persist: a found tape-win demos the loss writer */
        vfft_plan p = vfft_create(&cfg);
        if (!p)
            fprintf(stderr, "[natip] N=%d sample %d create FAILED\n", N, s);
        if (p)
            vfft_destroy(p);
        vfft_wisdom_save(W, dir);
        vfft_wisdom_free(W);
    }
    return 0;
}
