#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int NS[] = { 2048, 4096, 8192, 16384 };
    for (int p = 0; p < 2; p++)
      for (int i = 0; i < 4; i++) {
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C;
        c.placement = p ? VFFT_INPLACE : VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = NS[i]; c.howmany = 1;
        c.order = VFFT_ORDER_NATURAL; c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        vfft_plan pl = vfft_create(&c);
        fprintf(stderr, "== %s N=%d: %s\n", p ? "IP" : "OOP", NS[i],
                pl ? "served" : "REFUSED");
        if (pl) vfft_destroy(pl);
      }
    return 0;
}
