/* _adv_oddroute_reach2.c - which (layout,N) actually reach the ODD-MID
 * commit-site route race?  Prints the fingerprint; VFFT_ZT_LOG names the race. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
static const int NS[] = {2304,2560,3072,3584,5120,6144,7168,9216,10240,12288,20480,24576};
#define NN ((int)(sizeof NS/sizeof NS[0]))
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc > 1 && !strcmp(argv[1],"--list")) {
        for (i=0;i<2*NN;i++) printf("%2d %s.%d\n", i, (i<NN)?"sp":"il", NS[i%NN]);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1],"--cell")) { printf("usage: --cell <0..%d>\n",2*NN-1); return 2; }
    i = atoi(argv[2]); if (i<0||i>=2*NN) return 2;
    memset(&cfg,0,sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = (i<NN)?VFFT_LAYOUT_SPLIT:VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.dims = 1; cfg.n[0] = NS[i%NN]; cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s.%d\n", (i<NN)?"sp":"il", NS[i%NN]);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
