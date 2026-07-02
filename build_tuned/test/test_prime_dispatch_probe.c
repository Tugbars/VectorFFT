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

/* execute + roundtrip the dispatch plan directly (the exact path 2D uses: no bluestein wisdom). */
static void rt_probe(int N, size_t K)
{
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, &REG, NULL);
    if (!p) { printf("rt N=%d K=%-3zu  plan NULL\n", N, K); return; }
    size_t tot = (size_t)N * K;
    double *re=malloc(tot*8),*im=malloc(tot*8),*xr=malloc(tot*8),*xi=malloc(tot*8);
    srand(5+N+(int)K);
    for (size_t i=0;i<tot;i++){ double a=(double)rand()/RAND_MAX-0.5,b=(double)rand()/RAND_MAX-0.5; re[i]=xr[i]=a; im[i]=xi[i]=b; }
    vfft_proto_execute_fwd(p, re, im, K);
    vfft_proto_execute_bwd(p, re, im, K);
    double rt=0, inv=1.0/(double)N;
    for (size_t i=0;i<tot;i++){ double dr=fabs(re[i]*inv-xr[i]),di=fabs(im[i]*inv-xi[i]); if(dr>rt)rt=dr; if(di>rt)rt=di; }
    printf("rt N=%d K=%-3zu  roundtrip=%.2e %s\n", N, K, rt, rt<1e-9?"ok":"*** WRONG ***");
    free(re);free(im);free(xr);free(xi); stride_plan_destroy(p);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_init(&REG);
    probe(13, 8);
    probe(127, 100);
    printf("--- 1D dispatch roundtrip (the exact uncalibrated path 2D uses) ---\n");
    rt_probe(127, 8); rt_probe(127, 99); rt_probe(127, 100); rt_probe(127, 104); rt_probe(127, 128);
    return 0;
}
