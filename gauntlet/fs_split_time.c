/* fs_split_time.c -- the 1D c2c four-step at M, scrambled class (the plane as is) vs natural
 * (plus the ordering transpose), both directions, out of place: the ordering pass's cost. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "vfft.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#include <windows.h>
typedef struct { vfft_plan h; vfft_dir_t dir; double *in, *out; } arm_t;
static void arm_run(void *v) { arm_t *a = (arm_t *)v; vfft_execute(a->h, a->dir, a->in, NULL, a->out, NULL); }
static vfft_plan mk(vfft_wisdom *W, int N, int nat)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
    return vfft_create(&cfg);
}
int main(int argc, char **argv)
{
    SetThreadAffinityMask(GetCurrentThread(), 0x4); SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    for (int i = 2; i < argc; i++)
    {
        const int M = atoi(argv[i]);
        double *a = (double *)_aligned_malloc(2 * (size_t)M * 8, 64), *b = (double *)_aligned_malloc(2 * (size_t)M * 8, 64);
        for (size_t k = 0; k < 2 * (size_t)M; k++) a[k] = (double)(k % 977) * 1e-3 - 0.4;
        vfft_plan hs = mk(W, M, 0), hn = mk(W, M, 1);
        if (!hs || !hn) { printf("M=%d: plan NULL\n", M); continue; }
        arm_t s0 = { hs, VFFT_FORWARD, a, b }, n0 = { hn, VFFT_FORWARD, a, b }, s1 = { hs, VFFT_BACKWARD, a, b }, n1 = { hn, VFFT_BACKWARD, a, b };
        const vfft_race_arm_t arms[4] = { { "scr fwd", arm_run, &s0 }, { "nat fwd", arm_run, &n0 }, { "scr bwd", arm_run, &s1 }, { "nat bwd", arm_run, &n1 } };
        const vfft_race_proto_t proto = { 9, 1, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        double ns[4];
        vfft_race_run(&proto, arms, 4, ns);
        printf("c2c M=%-8d fwd: scrambled %.0f us, natural %.0f us (ordering pass %.0f) | bwd: scrambled %.0f, natural %.0f (ordering pass %.0f)  [%s | %s]\n",
               M, ns[0] / 1e3, ns[1] / 1e3, (ns[1] - ns[0]) / 1e3, ns[2] / 1e3, ns[3] / 1e3, (ns[3] - ns[2]) / 1e3, vfft_plan_route(hs), vfft_plan_route(hn));
    }
    return 0;
}
