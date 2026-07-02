/* test_prime_dispatch_probe.c — why does vfft_proto_auto_plan_dispatch build small primes (13) but
 * return NULL for large primes (127) with NO bluestein wisdom set? Probe Rader (M=N-1) + Bluestein
 * (heuristic M) inner builds directly. Build: python build.py --src test/test_prime_dispatch_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include "executor.h"
#include "planner.h"
#include "prime_dispatch.h"

static vfft_proto_registry_t REG;

static void probe(int N, size_t K)
{
    int prime = _vfft_is_prime(N);
    int rs = prime ? _vfft_is_radix_smooth(N - 1) : 0;
    printf("N=%-4d K=%-3zu prime=%d radix_smooth(N-1=%d)=%d\n", N, K, prime, N-1, rs);

    /* dispatch (no bluestein wisdom) */
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, &REG, NULL);
    printf("   auto_plan_dispatch(no-wis) -> %s\n", p ? "OK" : "NULL");
    if (p) stride_plan_destroy(p);

    if (prime) {
        /* Rader inner: CT(N-1) */
        int nm1 = N - 1;
        size_t B = _bluestein_block_size(nm1, K);
        stride_plan_t *ri = vfft_proto_auto_plan(nm1, B, &REG, NULL);
        printf("   Rader inner  CT(%d, B=%zu) -> %s\n", nm1, B, ri ? "OK" : "NULL");
        if (ri) stride_plan_destroy(ri);
        /* Bluestein inner: CT(chosen M) */
        int M = _bluestein_choose_m(N);
        size_t Bb = _bluestein_block_size(M, K);
        stride_plan_t *bi = vfft_proto_auto_plan(M, Bb, &REG, NULL);
        printf("   Bluestein inner CT(M=%d, B=%zu) -> %s\n", M, Bb, bi ? "OK" : "NULL");
        if (bi) stride_plan_destroy(bi);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_init(&REG);
    probe(13, 8);
    probe(127, 8);
    probe(127, 100);
    probe(251, 8);
    return 0;
}
