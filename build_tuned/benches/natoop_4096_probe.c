/* natoop_4096_probe.c — the minimal cold repro of vfft_natural_front_gate's
 * failing cell, S samples in one process.
 *
 * One sample = the gate's exact flow at one N on a FRESH caller-owned wisdom
 * bundle: a SCRAMBLED OOP create (banks the kind-4 cascade line the natoop
 * candidate replays) followed by a NATURAL OOP create (fires the natoop race:
 * cascade vs the K=1 OOP engine). VFFT_NAT_LOG=1 makes the race print
 *   [natorder] N=.. K=1 OOP zcasc=..ns engine=..ns -> ZCASC-OOP|engine
 * to stderr — one line per sample is the whole result.
 *
 * usage: probe [N=4096] [samples=5] [wisdir-base=probe_wis]
 * build: gcc -O2 -mavx2 -mfma <include flags> natoop_4096_probe.c src/core/vfft.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 4096;
    int S = argc > 2 ? atoi(argv[2]) : 5;
    const char *base = argc > 3 ? argv[3] : "probe_wis";
    static char envbuf[] = "VFFT_NAT_LOG=1";
    putenv(envbuf);
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    for (int s = 0; s < S; s++)
    {
        char dir[512];
        if (S == 1) snprintf(dir, sizeof dir, "%s", base); /* exact dir: bank into a real store */
        else        snprintf(dir, sizeof dir, "%s_%d", base, s);
        vfft_wisdom *W = vfft_wisdom_load(dir); /* empty dir = fresh bundle */
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.dims = 1;
        cfg.n[0] = N;
        cfg.howmany = 1;
        cfg.rigor = VFFT_MEASURE;
        cfg.wisdom = W;
        cfg.wisdom_write = 1; /* banking run: verdicts persist */
        cfg.order = VFFT_ORDER_SCRAMBLED;
        fprintf(stdout, "sample %d:\n", s);
        vfft_plan p1 = vfft_create(&cfg);
        if (!p1)
        {
            fprintf(stdout, "  scrambled create FAILED\n");
            vfft_wisdom_free(W);
            continue;
        }
        cfg.order = VFFT_ORDER_NATURAL;
        vfft_plan p2 = vfft_create(&cfg); /* fires the natoop race */
        if (!p2)
            fprintf(stdout, "  natural create FAILED\n");
        if (p2)
            vfft_destroy(p2);
        vfft_destroy(p1);
        vfft_wisdom_save(W, dir);
        vfft_wisdom_free(W);
    }
    return 0;
}
