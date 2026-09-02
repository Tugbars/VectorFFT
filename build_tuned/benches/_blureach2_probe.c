#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
typedef struct { const char *tag; int t, lay, place, ord, n0, n1; } cell_t;
#define IL VFFT_LAYOUT_INTERLEAVED
#define IP VFFT_INPLACE
#define OP VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
static const cell_t CELLS[] = {
  {"2d.il.oop.c2c.13x100",  VFFT_C2C,IL,OP,DEF, 13,100},   /* prime N1 IN the pool */
  {"2d.il.oop.c2c.19x100",  VFFT_C2C,IL,OP,DEF, 19,100},   /* prime N1 IN the pool */
  {"2d.il.oop.c2c.254x100", VFFT_C2C,IL,OP,DEF,254,100},   /* composite 2*127, unexpressible */
  {"2d.il.oop.c2c.128x100", VFFT_C2C,IL,OP,DEF,128,100},   /* pow2 but 128=32*4 chain */
  {"2d.il.oop.r2c.127x256", VFFT_R2C,IL,OP,DEF,127,256},   /* REAL tier, prime N1 */
  {"2d.il.oop.c2r.127x256", VFFT_C2R,IL,OP,DEF,127,256},
  {"2d.il.ip.r2c.127x256",  VFFT_R2C,IL,IP,DEF,127,256},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))
int main(int argc, char **argv)
{
    static char buf[65536]; long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p; int i;
    if (argc < 3) return 2;
    i = atoi(argv[2]); if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)CELLS[i].t;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout = (vfft_layout_t)CELLS[i].lay;
    cfg.order = CELLS[i].ord; cfg.dims = 2;
    cfg.n[0] = CELLS[i].n0; cfg.n[1] = CELLS[i].n1;
    cfg.howmany = 1; cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s\n", CELLS[i].tag);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout);
    vfft_destroy(p); return 0;
}
