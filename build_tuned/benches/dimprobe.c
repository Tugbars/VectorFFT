#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static void tryc(vfft_wisdom *W, int tr, int lay, int n1, int n2) {
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = tr; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = n1; c.n[1] = n2;
    c.howmany = 1; c.layout = lay;
    c.wisdom = W; c.wisdom_write = 0;
    vfft_plan p = vfft_create(&c);
    const char *tn = tr == VFFT_C2C ? "c2c" : (tr == VFFT_R2C ? "r2c" : "c2r");
    if (!p) { printf("%s %s %dx%d: REFUSED (loud)\n",
                     tn, lay == VFFT_LAYOUT_INTERLEAVED ? "IL " : "SPL", n1, n2); return; }
    /* served: correctness spot-check via DC identity */
    size_t hp1 = (size_t)n2 / 2 + 1;
    size_t sn = tr == VFFT_R2C ? (size_t)n1 * n2 : 2 * (size_t)n1 * n2;
    size_t dn = tr == VFFT_R2C ? 2 * (size_t)n1 * hp1 : 2 * (size_t)n1 * n2;
    double *x = calloc(sn, 8), *z = calloc(dn > sn ? dn : sn, 8);
    double s0 = 0, s1 = 0;
    for (size_t i = 0; i < sn; i++) { x[i] = (double)rand() / RAND_MAX; }
    if (tr == VFFT_R2C) { for (size_t i = 0; i < sn; i++) s0 += x[i]; }
    else for (size_t i = 0; i < sn / 2; i++) { s0 += x[2*i]; s1 += x[2*i+1]; }
    vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
    double dc = fabs(z[0] - s0) + fabs(z[1] - s1);
    printf("%s %s %dx%d: SERVED, dc-err %.1e %s\n",
           tn, lay == VFFT_LAYOUT_INTERLEAVED ? "IL " : "SPL", n1, n2, dc,
           dc < 1e-8 * (s0 > 1 ? s0 : 1) ? "OK" : "*** SILENTLY WRONG ***");
    vfft_destroy(p); free(x); free(z);
}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    tryc(W, VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, 127, 100);
    tryc(W, VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, 127, 128);
    tryc(W, VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, 128, 127);
    tryc(W, VFFT_C2C, VFFT_LAYOUT_SPLIT, 127, 100);
    tryc(W, VFFT_R2C, VFFT_LAYOUT_INTERLEAVED, 127, 100);
    tryc(W, VFFT_R2C, VFFT_LAYOUT_INTERLEAVED, 128, 254);
    tryc(W, VFFT_R2C, VFFT_LAYOUT_INTERLEAVED, 127, 128);
    return 0;
}
