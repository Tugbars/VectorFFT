/* ip_scr_probe.c -- does an IN-PLACE SCRAMBLED create at a solo-kernel N
 * refuse on a cold store? (section B2, 2026-09-17). The in-place door's
 * mono candidate reads the ord=NAT row unconditionally; a scrambled cell
 * whose own race picked mono may find nothing to build and refuse.
 * Run: ip_scr_probe.exe <cold wisdir> <N> [nat=1]   (VFFT_NAT_LOG=1 shows the engine) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    const int N = argc > 2 ? atoi(argv[2]) : 32;
    const int nat = argc > 3 ? atoi(argv[3]) : 0;
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE; cfg.rigor = VFFT_PATIENT;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.nthreads = 1; cfg.wisdom = W; cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    printf("N=%d in-place %s: %s\n", N, nat ? "natural" : "scrambled", p ? "created" : "REFUSED");
    if (p) vfft_destroy(p);
    vfft_wisdom_free(W);
    return p ? 0 : 1;
}
