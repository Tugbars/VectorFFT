/* door_time.c -- how long the real door takes to CREATE a plan at N (out of
 * place, recalibrate = the race runs and banks), both directions, on a
 * scratch store; VFFT_ZRACE_VERBOSE=1 shows the engine race, VFFT_NAT_LOG=1
 * the ZTT's refusals. Build: python gauntlet/build.py --compile --vfft --src gauntlet/door_time.c
 * Run: door_time <scratch store> N [N ...] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "common/support/race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 3) { fprintf(stderr, "usage: door_time <scratch store> N [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    for (int a = 2; a < argc; a++)
    {
        const int N = atoi(argv[a]);
        for (int c2r = 0; c2r < 2; c2r++)
        {
            vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
            cfg.transform = c2r ? VFFT_C2R : VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
            cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
            cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
            cfg.recalibrate = getenv("VFFT_DOOR_RECAL") ? atoi(getenv("VFFT_DOOR_RECAL")) : 1;   /* VFFT_DOOR_RECAL=0: a plain create (a miss races, a hit replays) */
            double t0 = vfft_now_ns();
            vfft_plan h = vfft_create(&cfg);
            double t1 = vfft_now_ns();
            printf("N=%-6d %s create %.0f ms %s\n", N, c2r ? "c2r" : "r2c", (t1 - t0) / 1e6, h ? "" : "(NULL)");
            fflush(stdout);
            if (h) vfft_destroy(h);
        }
    }
    return 0;
}
