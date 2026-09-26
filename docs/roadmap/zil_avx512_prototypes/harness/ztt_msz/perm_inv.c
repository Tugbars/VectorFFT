/* dp_planner_il.h:790-812 inverts the plain perm with "the lane order is its
 * own inverse" (sig = {0,2,1,3}). At VW=8 sigma = {0,4,1,5,2,6,3,7} is NOT an
 * involution: check the planner's inversion (generalised naively) vs sigma^-1. */
#include <stdio.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
int main(void)
{
    const int N = 4096, ch[4] = { 8, 8, 8, 8 };
    vfft_ztt_plan_t *p = _ztt_create(N, ch, 4, 1, 1);
    const long VW = 8, R = 8; long idx, bad_naive = 0, bad_inv = 0; int i, rch[4];
    long sig[8], inv[8]; for (i = 0; i < 8; i++) { sig[i] = (i >> 1) + (i & 1) * 4; inv[sig[i]] = i; }
    for (i = 0; i < 4; i++) rch[i] = ch[3 - i];
    for (idx = 0; idx < N; idx++) {
        const long span = idx / (VW * R), within = idx % (VW * R), pp = within / VW, j = within % VW;
        const long k_naive = _ztt_digitrev((VW * span + sig[j]) * R + pp, rch, 4);
        const long k_inv   = _ztt_digitrev((VW * span + inv[j]) * R + pp, rch, 4);
        if (vfft_ztt_perm(p, k_naive) != (size_t)idx) bad_naive++;
        if (vfft_ztt_perm(p, k_inv) != (size_t)idx) bad_inv++;
    }
    printf("N=%d chain 8.8.8.8 plain, VW=8: planner-style inverse with sigma: %ld/%d positions wrong; with sigma^-1: %ld wrong\n", N, bad_naive, N, bad_inv);
    return 0;
}
