/* _adv_2dnat_split.c — does SPLIT 2D c2c order=NATURAL reach the J_nat sweep
 * and bank an ord=nat 2D record?  One process, one cell. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    int N   = (argc > 1) ? atoi(argv[1]) : 64;
    int lay = (argc > 2) ? atoi(argv[2]) : 0;   /* 0=split 1=il */
    char buf[4096];
    long ctr[VFFT__FP_NCOUNTERS];
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = (argc > 3 && atoi(argv[3])) ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    c.layout    = lay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    c.order     = VFFT_ORDER_NATURAL;
    c.rigor     = VFFT_MEASURE;
    c.dims      = 2;
    c.n[0] = N; c.n[1] = N;
    c.howmany   = 1;
    c.nthreads  = 1;
    c.wisdom_write = 1;
    vfft_plan p = vfft_create(&c);
    printf("create %dx%d %s c2c NAT -> %s\n", N, N,
           lay ? "IL" : "SPLIT", p ? "ACCEPT" : "REFUSED");
    vfft__fp_counters(ctr);
    printf("races=%ld\n", ctr[5]);
    if (p) {
        vfft__fingerprint(p, buf, sizeof buf);
        printf("fp: %s\n", buf);
        vfft_destroy(p);
    }
    return 0;
}
