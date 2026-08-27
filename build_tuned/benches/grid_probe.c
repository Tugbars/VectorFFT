#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const struct { int n; const char *cls; } NS[] = {
        { 1024, "pow2" },  { 63, "smooth odd" }, { 129, "awkward 3*43" },
        { 101, "prime" },  { 4099, "big prime" },
    };
    static const int PL[] = { VFFT_OUTOFPLACE, VFFT_INPLACE };
    static const int OR[] = { 0 /*DEFAULT*/, VFFT_ORDER_NATURAL,
                              VFFT_ORDER_SCRAMBLED };
    static const char *on[] = { "dflt", "nat ", "scr " };
    printf("%-14s %-8s dflt   nat    scr\n", "N", "");
    for (int i = 0; i < 5; i++) {
        for (int p = 0; p < 2; p++) {
            printf("%-6d %-7s %-4s ", NS[i].n, NS[i].cls,
                   p ? "IP " : "OOP");
            for (int o = 0; o < 3; o++) {
                vfft_config_t c; memset(&c, 0, sizeof c);
                c.transform = VFFT_C2C; c.placement = PL[p];
                c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = NS[i].n;
                c.howmany = 1; c.order = OR[o];
                c.layout = VFFT_LAYOUT_INTERLEAVED;
                c.wisdom = W; c.wisdom_write = 0;
                vfft_plan pl = vfft_create(&c);
                printf("  %-5s", pl ? "yes" : "NO");
                if (pl) vfft_destroy(pl);
            }
            printf("\n");
        }
    }
    return 0;
}
