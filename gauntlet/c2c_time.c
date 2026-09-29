/* c2c_time.c -- how long the c2c door takes to CREATE a plan at N under
 * recalibrate, the four cases the zr2c child can be (fwd/bwd x oop natural /
 * in-place natural), on a scratch store. Build: python gauntlet/build.py --compile --vfft --src gauntlet/c2c_time.c
 * Run: c2c_time <scratch store> N [N ...] */
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
    if (argc < 3) { fprintf(stderr, "usage: c2c_time <scratch store> N [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    for (int a = 2; a < argc; a++)
    {
        const int N = atoi(argv[a]);
        for (int ip = 0; ip < 2; ip++)
            for (int recal = 0; recal < 2; recal++)
            {
                vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
                cfg.transform = VFFT_C2C; cfg.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
                cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
                cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_NATURAL;
                cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1; cfg.recalibrate = recal;
                double t0 = vfft_now_ns();
                vfft_plan h = vfft_create(&cfg);
                double t1 = vfft_now_ns();
                printf("N=%-6d c2c %s recal=%d create %.0f ms  route %s\n", N, ip ? "IP " : "OOP", recal,
                       (t1 - t0) / 1e6, h && vfft_plan_route(h) ? vfft_plan_route(h) : "?");
                fflush(stdout);
                if (h) vfft_destroy(h);
            }
    }
    return 0;
}
