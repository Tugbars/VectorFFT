/* adversarial reach probe #2: does the il2d wl/roop axis race actually
 * fire at the claimed odd cells 45x64 and 64x63? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;
#define C2C VFFT_C2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"A.2d.il.oop.c2c.45x64.def",  C2C,IL,OP,DEF,2, 45, 64,1},
  {"B.2d.il.ip.c2c.45x64.def",   C2C,IL,IP,DEF,2, 45, 64,1},
  {"C.2d.il.oop.c2c.64x63.def",  C2C,IL,OP,DEF,2, 64, 63,1},
  {"D.2d.il.ip.c2c.64x63.def",   C2C,IL,IP,DEF,2, 64, 63,1},
  {"E.2d.il.oop.c2c.45x64.nat",  C2C,IL,OP,NAT,2, 45, 64,1},
  {"F.2d.il.oop.c2c.127x64.def", C2C,IL,OP,DEF,2,127, 64,1},
  {"G.2d.il.oop.c2c.15x64.def",  C2C,IL,OP,DEF,2, 15, 64,1},
  {"H.2d.il.oop.c2c.64x45.def",  C2C,IL,OP,DEF,2, 64, 45,1},
  {"I.2d.il.oop.c2c.64x64.def",  C2C,IL,OP,DEF,2, 64, 64,1},
  {"J.2d.il.oop.c2c.45x63.def",  C2C,IL,OP,DEF,2, 45, 63,1},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc > 1 && !strcmp(argv[1], "--list")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].tag);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage --cell i\n"); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)CELLS[i].t;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = CELLS[i].dims;
    cfg.n[0] = CELLS[i].n0; cfg.n[1] = CELLS[i].n1;
    cfg.howmany = CELLS[i].K;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
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
