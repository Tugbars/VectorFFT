/* reach probe: does 1D c2c OOP INTERLEAVED K=1 at odd/prime/awkward N
 * reach the K=1 IL route ladder (vfft.c 8408/8495/8508)? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int n; int place; int lay; } cell_t;
static const cell_t CELLS[] = {
  {"oop.il.256   (control pow2)", 256, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.45    (odd pair 9x5)",  45, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.3072  (odd*2^k)",     3072, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.127   (prime)",        127, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.129   (awkward 3*43)", 129, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.1009  (prime)",       1009, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.sp.45    (odd, split)",    45, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT},
  {"oop.sp.127   (prime, split)",  127, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT},
  {"ip.il.127    (prime, inpl)",  127, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.192   (odd*2^k sm)",   192, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.1792  (odd*2^k)",     1792, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
  {"oop.il.50    (2*odd 5x10)",    50, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED},
};
#define NC ((int)(sizeof CELLS/sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc>1 && !strcmp(argv[1],"--list")) {
        for (i=0;i<NC;i++) printf("%2d %s\n", i, CELLS[i].tag);
        return 0;
    }
    if (argc<3 || strcmp(argv[1],"--cell")) { printf("usage: --cell <0..%d>|--list\n",NC-1); return 2; }
    i = atoi(argv[2]);
    if (i<0||i>=NC) { printf("range\n"); return 2; }
    memset(&cfg,0,sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 1;
    cfg.n[0]      = CELLS[i].n;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s\n", CELLS[i].tag);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
