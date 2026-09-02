#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
static const int NS[] = {3,5,7,9,11,13,15,21,25,27,33,35,45,49,51,55,63,
                         65,75,77,85,91,99,115,119,121,125,129,143,169,
                         187,189,202,203,209,221,247,253,255,289,323,361,
                         391,403,437,481,529,551,589,667,713,841,899,961,0};
int main(void)
{
    int i; vfft_config_t cfg; vfft_plan p;
    printf("in-place K=1 IL c2c NATURAL refusals (the il2d_rof trigger):\n");
    for (i = 0; NS[i]; i++) {
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_NATURAL;
        cfg.dims = 1; cfg.n[0] = NS[i]; cfg.howmany = 1; cfg.rigor = VFFT_MEASURE;
        p = vfft_create(&cfg);
        printf("  N=%-5d %s\n", NS[i], p ? "ok" : "REFUSE");
        fflush(stdout);
        if (p) vfft_destroy(p);
    }
    return 0;
}
