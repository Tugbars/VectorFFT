/* ilprime_composite_probe.c — does the prime engine build for a COMPOSITE N?
 * (2026-09-19, the owner's question: "can't our bluestein order have an answer
 * for it though" — 3007 = 31*97 is refused by the 1D door.)
 *
 * Bluestein needs no primality: M = the next power of two >= 2N - 1, chirp,
 * convolve, chirp. vfft_ilprime_create_method already routes a composite N
 * straight to _ilprime_create_bluestein. So the probe asks, per N: does the
 * plan build, what M, and which inner did the structural rule pick?
 *
 * Run:   ilprime_composite_probe.exe [N ...]
 * Build: python build.py --compile --vfft --src benches/ilprime_composite_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "wisdom2.h"
#include "oop/ztt.h"
#include "oop/k1_fourstep_band.h"
#include "planning/policy.h"
#include "support/race.h"   /* il_prime.h races its two methods */
#include "oop/il_prime.h"

static void one(int N)
{
    int M = 16;
    _ilprime_inner_t in;
    vfft_ilprime_plan_t *p;
    int pair_ok = 0, R2, cR2 = 0, cA = 0, cB = 0, chain_ok;
    while (M < 2 * N - 1) M <<= 1;
    /* the structural rule's own two steps, spelled out so the failure is named */
    for (R2 = (M < 64 ? M : 64); R2 >= 3; R2--)
    {
        int R1;
        if (M % R2) continue;
        R1 = M / R2;
        if (R1 < 3 || R1 > 64) continue;
        if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
        pair_ok = 1;
        break;
    }
    chain_ok = vfft_il3p_default_chain(M, &cR2, &cA, &cB);
    memset(&in, 0, sizeof in);
    p = vfft_ilprime_create(N);
    printf("N=%-8d %-9s M=%-8d pair=%-3s chain3=%-3s", N,
           _ilprime_is_prime(N) ? "prime" : "composite", M,
           pair_ok ? "yes" : "NO", chain_ok ? "yes" : "NO");
    if (chain_ok) printf(" (%d.%d.%d)", cR2, cA, cB);
    printf("  -> plan %s\n", p ? "BUILT" : "*** NULL ***");
    if (p) vfft_ilprime_destroy(p);
}

int main(int argc, char **argv)
{
    static const int def[] = { 3007, 3013, 3002, 3005, 3017, 2003, 1021, 4099, 8191, 6005, 12005 };
    int i;
    if (argc > 1)
        for (i = 1; i < argc; i++) one(atoi(argv[i]));
    else
        for (i = 0; i < (int)(sizeof def / sizeof def[0]); i++) one(def[i]);
    return 0;
}
